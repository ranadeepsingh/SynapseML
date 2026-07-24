// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.azure.synapse.ml.nbtest

import org.scalatest.funsuite.AnyFunSuite

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
}
