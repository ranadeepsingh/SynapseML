// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.azure.synapse.ml.nbtest

import org.scalatest.funsuite.AnyFunSuite
import spray.json.DefaultJsonProtocol._
import spray.json._

import java.time.Instant

class DatabricksUtilitiesSuite extends AnyFunSuite {

  test("Reuse tokens with more than five minutes remaining") {
    val now = Instant.parse("2026-07-24T12:00:00Z")

    assert(DatabricksUtilities.hasSufficientTokenLifetime(now.plusSeconds(301), now))
  }

  test("Refresh tokens at the five minute buffer") {
    val now = Instant.parse("2026-07-24T12:00:00Z")

    assert(!DatabricksUtilities.hasSufficientTokenLifetime(now.plusSeconds(300), now))
    assert(!DatabricksUtilities.hasSufficientTokenLifetime(now.minusSeconds(1), now))
  }

  test("Use separate worker and driver pools for GPU clusters") {
    val initScripts = """[{"dbfs":{"destination":"dbfs:/init.sh"}}]"""
    val request = DatabricksUtilities.createClusterRequest(
      "gpu-cluster",
      "gpu-runtime",
      2,
      "gpu-pool",
      initScripts = initScripts,
      driverInstancePoolId = Some("cpu-pool")
    ).parseJson.asJsObject

    assert(request.fields("instance_pool_id").convertTo[String] === "gpu-pool")
    assert(request.fields("driver_instance_pool_id").convertTo[String] === "cpu-pool")
    assert(request.fields("init_scripts") === initScripts.parseJson)
  }

  test("Omit separate driver pool by default") {
    val request = DatabricksUtilities.createClusterRequest(
      "cpu-cluster",
      "cpu-runtime",
      5,
      "cpu-pool"
    ).parseJson.asJsObject

    assert(!request.fields.contains("driver_instance_pool_id"))
  }

  test("Select all GPU notebooks in deterministic order") {
    val notebookNames = DatabricksUtilities.GPUNotebooks.map(_.getName)

    assert(notebookNames === Seq(
      "Quickstart - Apply Phi Model with HuggingFace CausalLM.ipynb",
      "Quickstart - Fine-tune a Text Classifier.ipynb",
      "Quickstart - Fine-tune a Vision Classifier.ipynb"
    ))
  }
}
