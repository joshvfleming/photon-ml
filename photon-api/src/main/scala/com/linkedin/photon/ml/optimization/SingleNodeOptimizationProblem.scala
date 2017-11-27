/*
 * Copyright 2017 LinkedIn Corp. All rights reserved.
 * Licensed under the Apache License, Version 2.0 (the "License"); you may
 * not use this file except in compliance with the License. You may obtain a
 * copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 */
package com.linkedin.photon.ml.optimization

import scala.util.Random

import breeze.linalg.{DenseVector, Vector}
import org.apache.spark.broadcast.Broadcast

import com.linkedin.photon.ml.constants.MathConst
import com.linkedin.photon.ml.data.LabeledPoint
import com.linkedin.photon.ml.evaluation.Evaluator
import com.linkedin.photon.ml.function._
import com.linkedin.photon.ml.model.Coefficients
import com.linkedin.photon.ml.normalization.NormalizationContext
import com.linkedin.photon.ml.optimization.game.GLMOptimizationConfiguration
import com.linkedin.photon.ml.supervised.model.{GeneralizedLinearModel, ModelTracker}

import com.linkedin.photon.ml.Types.REType
import com.linkedin.photon.ml.hyperparameter.EvaluationFunction
import com.linkedin.photon.ml.hyperparameter.search.{GaussianProcessSearch, RandomSearch}

/**
 * An optimization problem solved by a single task on one executor. Used for solving the per-entity optimization
 * problems of a random effect model.
 *
 * @tparam Objective The objective function to optimize, using a single node
 * @param optimizer The underlying optimizer which iteratively solves the convex problem
 * @param objectiveFunction The objective function to optimize
 * @param glmConstructor The function to use for producing GLMs from trained coefficients
 * @param isComputingVariances Should coefficient variances be computed in addition to the means?
 * @param tuningEvaluator Evaluator used for hyperparameter tuning
 * @param seed Randoms seed
 */
protected[ml] class SingleNodeOptimizationProblem[Objective <: SingleNodeObjectiveFunction] protected[optimization] (
    optimizer: Optimizer[Objective],
    objectiveFunction: Objective,
    glmConstructor: Coefficients => GeneralizedLinearModel,
    isComputingVariances: Boolean,
    tuningEvaluator: Option[Evaluator] = None,
    tuningSampleLowerBound: Int = Int.MaxValue,
    tuningRange: (Double, Double) = (1e-3, 1e4),
    tuningIterations: Int = 0,
    randomEffectType: REType = "none",
    seed: Long = System.currentTimeMillis)
  extends GeneralizedLinearOptimizationProblem[Objective](
    optimizer,
    objectiveFunction,
    glmConstructor,
    isComputingVariances)
  with Serializable {

  val rand = new Random(seed)

  /**
   * Compute coefficient variances
   *
   * @param input The training data
   * @param coefficients The feature coefficients means
   * @return The feature coefficient variances
   */
  override def computeVariances(input: Iterable[LabeledPoint], coefficients: Vector[Double]): Option[Vector[Double]] = {
    (isComputingVariances, objectiveFunction) match {
      case (true, twiceDiffFunc: TwiceDiffFunction) =>
        Some(twiceDiffFunc
          .hessianDiagonal(input, coefficients)
          .map(v => 1.0 / (v + MathConst.EPSILON)))

      case _ =>
        None
    }
  }

  /**
   * Run the optimization algorithm on the input data, starting from an initial model of all-0 coefficients.
   *
   * @param input The training data
   * @return The learned GLM for the given optimization problem, data, regularization type, and regularization weight
   */
  override def run(input: Iterable[LabeledPoint]): GeneralizedLinearModel =
    run(input, initializeZeroModel(input.head.features.size))

  /**
   * Evaluation function to use for single-node optimization problem hyperparameter tuning
   *
   * TODO move this to a separate file
   *
   * @param optimizer The optimizer
   * @param objectiveFunction The objective function
   * @param initialModel The initial model
   * @param trainInput The training set
   * @param validationInput The validation set
   */
  class SingleNodeEvaluationFunction(
      optimizer: Optimizer[Objective],
      objectiveFunction: Objective,
      initialModel: GeneralizedLinearModel,
      trainInput: Iterable[LabeledPoint],
      validationInput: Iterable[LabeledPoint])
    extends EvaluationFunction[(Double, Double)] {

    /**
      * Performs the evaluation
      *
      * @param hyperParameters the vector of hyperparameter values under which to evaluate the function
      * @return a tuple of the evaluated value and the original output from the inner estimator
      */
    override def apply(hyperParameters: DenseVector[Double]): (Double, (Double, Double)) = {
      // Unpack and update regularization weight
      val regularizationWeight = hyperParameters(0)
      objectiveFunction match {
        case func: L2Regularization => func.l2RegularizationWeight = regularizationWeight
      }

      // Train using the new regularization weight
      val normalizationContext = optimizer.getNormalizationContext
      val (optimizedCoefficients, _) = optimizer.optimize(objectiveFunction, initialModel.coefficients.means)(trainInput)
      val optimizedVariances = computeVariances(trainInput, optimizedCoefficients)
      val model = createModel(normalizationContext, optimizedCoefficients, optimizedVariances)

      // Score the validation set with the new model
      val scoresLabelsAndWeights = validationInput
        .map(x => (0L, (model.computeMean(x.features, x.offset), x.label, x.weight)))

      // Evaluate the validation scores
      val evaluation = tuningEvaluator.map { evaluator =>
        evaluator.evaluateWithScoresAndLabelsAndWeights(scoresLabelsAndWeights)
      }.getOrElse(0.0)

      (evaluation, (regularizationWeight, evaluation))
    }

    /**
      * Extracts a vector representation from the hyperparameters associated with the original estimator output
      *
      * @param result the original estimator output
      * @return vector representation
      */
    override def vectorizeParams(result: (Double, Double)): DenseVector[Double] =
      DenseVector(result._1)

    /**
      * Extracts the evaluated value from the original estimator output
      *
      * @param result the original estimator output
      * @return the evaluated value
      */
    override def getEvaluationValue(result: (Double, Double)): Double = result._2
  }

  /**
   * Runs hyperparameter optimization to find the best regularization weight, given the inputs
   *
   * @param input The original training data
   * @param initialModel The initial model
   * @param objectiveFunction The objective function
   * @param evaluator The evaluator to use for hyperparameter metrics and selection
   * @param trainingSetSplit The fraction of input data to use for training set during hyperparameter optimization. The
   *   rest is held back as a validation set.
   * @param range The range of regularization weights to explore
   * @param iterations The number of hyperparameter tuning iterations
   */
  def runHyperparameterTuning(
      input: Iterable[LabeledPoint],
      initialModel: GeneralizedLinearModel,
      initialWeight: Double,
      objectiveFunction: Objective,
      evaluator: Evaluator,
      trainingSetSplit: Double,
      range: (Double, Double),
      iterations: Int,
      positiveExampleLowerBound: Int = 1,
      negativeExampleLowerBound: Int = 1): Option[Double] = {

    val inputWithProb = input.map((rand.nextDouble, _))
    val trainData = inputWithProb.filter(_._1 < trainingSetSplit).map(_._2).toIterable
    val validationData = inputWithProb.filter(_._1 >= trainingSetSplit).map(_._2).toIterable

    if (trainData.count(_.label > 0) < positiveExampleLowerBound
      || trainData.count(_.label <= 0) < negativeExampleLowerBound
      || validationData.count(_.label > 0) < positiveExampleLowerBound
      || validationData.count(_.label <= 0) < negativeExampleLowerBound) {
      return None
    }

    val evaluationFunction = new SingleNodeEvaluationFunction(
      optimizer,
      objectiveFunction,
      initialModel,
      trainData,
      validationData)

    val (initialEval, _) = evaluationFunction(DenseVector(initialWeight))

    // This is hanging, for some reason. Each model train / test cycle happens really fast for these sub-problems,
    // though, so it might not be necessary since we can do a lot of evaluations
    // val searcher = new GaussianProcessSearch[(GeneralizedLinearModel, Double, Double)](
    //   List(range),
    //   evaluationFunction,
    //   evaluator,
    //   seed = seed)

    val searcher = new RandomSearch[(Double, Double)](
      List(range),
      evaluationFunction,
      seed = seed)

    val results = searcher.find(iterations)

    // TODO use evaluator.betterThan as a comparator instead
    val (bestWeight, bestEval) = results.maxBy(_._2)
    if (!bestEval.isNaN && !bestEval.isInfinite && bestEval > initialEval) {
      logger.info(s"@!!rehyper_imp_${randomEffectType}_${bestEval/initialEval - 1}")
      Some(bestWeight)
    } else {
      None
    }
  }

  /**
   * Run the optimization algorithm on the input data, starting from the initial model provided.
   *
   * @param input The training data
   * @param initialModel The initial model from which to begin optimization
   * @return The learned GLM for the given optimization problem, data, regularization type, and regularization weight
   */
  override def run(input: Iterable[LabeledPoint], initialModel: GeneralizedLinearModel): GeneralizedLinearModel = {
    // If there's a tuning evaluator, run hyperparameter tuning to find an optimal regularization weight
    tuningEvaluator match {
      case Some(evaluator) if (input.size >= tuningSampleLowerBound) =>
        objectiveFunction match {
          case func: L2Regularization =>
            val optimalRegWeight = runHyperparameterTuning(
              input,
              initialModel,
              func.l2RegularizationWeight,
              objectiveFunction,
              evaluator,
              0.8,
              tuningRange,
              tuningIterations)

            optimalRegWeight.foreach { weight =>
              logger.info(s"@!!rehyper_${randomEffectType}_${weight}")
              func.l2RegularizationWeight = weight
            }
        }

      case _ =>
    }

    val normalizationContext = optimizer.getNormalizationContext
    val (optimizedCoefficients, _) = optimizer.optimize(objectiveFunction, initialModel.coefficients.means)(input)
    val optimizedVariances = computeVariances(input, optimizedCoefficients)

    modelTrackerBuilder.foreach { modelTrackerBuilder =>
      val tracker = optimizer.getStateTracker.get
      logger.info(s"History tracker information:\n $tracker")
      val modelsPerIteration = tracker.getTrackedStates.map { x =>
        val coefficients = x.coefficients
        val variances = computeVariances(input, coefficients)
        createModel(normalizationContext, coefficients, variances)
      }
      logger.info(s"Number of iterations: ${modelsPerIteration.length}")
      modelTrackerBuilder += new ModelTracker(tracker, modelsPerIteration)
    }

    createModel(normalizationContext, optimizedCoefficients, optimizedVariances)
  }
}

object SingleNodeOptimizationProblem {
  /**
   * Factory method to create new SingleNodeOptimizationProblems.
   *
   * @param configuration The optimization problem configuration
   * @param objectiveFunction The objective function to optimize
   * @param glmConstructor The function to use for producing GLMs from trained coefficients
   * @param normalizationContext The normalization context
   * @param isTrackingState Should the optimization problem record the internal optimizer states?
   * @param isComputingVariance Should coefficient variances be computed in addition to the means?
   * @param tuningEvaluator Evaluator used for hyperparameter tuning
   * @return A new SingleNodeOptimizationProblem
   */
  def apply[Function <: SingleNodeObjectiveFunction](
      configuration: GLMOptimizationConfiguration,
      objectiveFunction: Function,
      glmConstructor: Coefficients => GeneralizedLinearModel,
      normalizationContext: Broadcast[NormalizationContext],
      isTrackingState: Boolean,
      isComputingVariance: Boolean,
      tuningEvaluator: Option[Evaluator] = None,
      tuningSampleLowerBound: Int,
      tuningRange: (Double, Double),
      tuningIterations: Int,
      randomEffectType: REType = "none"): SingleNodeOptimizationProblem[Function] = {

    val optimizerConfig = configuration.optimizerConfig
    val regularizationContext = configuration.regularizationContext
    val regularizationWeight = configuration.regularizationWeight
    // Will result in a runtime error if created Optimizer cannot be cast to an Optimizer that can handle the given
    // objective function.
    val optimizer = OptimizerFactory
      .build(optimizerConfig, normalizationContext, regularizationContext, regularizationWeight, isTrackingState)
      .asInstanceOf[Optimizer[Function]]

    new SingleNodeOptimizationProblem(
      optimizer,
      objectiveFunction,
      glmConstructor,
      isComputingVariance,
      tuningEvaluator,
      tuningSampleLowerBound,
      tuningRange,
      tuningIterations,
      randomEffectType)
  }
}
