// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.azure.synapse.ml.nbtest

import com.microsoft.azure.synapse.ml.nbtest.DatabricksUtilities._

class DatabricksGPUTests extends DatabricksTestHelper {

  private val gpuTimeoutMs = 30 * 60 * 1000
  // Reuse the scarce GPU workers sequentially while the driver runs from the CPU pool.
  val clusterId: String = createClusterInPool(
    GPUClusterName,
    AdbGpuRuntime,
    2,
    GpuPoolId,
    driverInstancePoolId = Some(PoolId)
  )

  databricksTestHelper(clusterId, GPULibraries, GPUNotebooks, 1, List(), gpuTimeoutMs)

  protected override def afterAll(): Unit = {
    afterAllHelper(clusterId, GPUClusterName)
    super.afterAll()
  }
}
