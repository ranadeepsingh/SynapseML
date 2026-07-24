// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.azure.synapse.ml

import org.scalatest.funsuite.AnyFunSuite
import spray.json.{JsNumber, JsString}

import java.time.Instant

class SecretsSuite extends AnyFunSuite {

  test("Parse Azure CLI access token expiry") {
    val token = Secrets.parseExpiringAccessToken(Map(
      "accessToken" -> JsString("test-token"),
      "expires_on" -> JsNumber(1777000000L)
    ))

    assert(token.value === "test-token")
    assert(token.expiresAt === Instant.ofEpochSecond(1777000000L))
  }

  test("Reject Azure CLI access token without numeric expiry") {
    val error = intercept[IllegalStateException] {
      Secrets.parseExpiringAccessToken(Map("accessToken" -> JsString("test-token")))
    }

    assert(error.getMessage === "Azure CLI access token response did not include expires_on")
  }
}
