#!/usr/bin/env bash

set -euo pipefail

echo "DEBUG: Getting GitHub OIDC token..." >&2
ID_TOKEN_RESPONSE=$(curl -sLS -H "User-Agent: actions/oidc-client" -H "Authorization: Bearer $ACTIONS_ID_TOKEN_REQUEST_TOKEN" \
    "${ACTIONS_ID_TOKEN_REQUEST_URL}&audience=${ARTIFACTORY_URL}")
echo "DEBUG: ID token response: $ID_TOKEN_RESPONSE" >&2

ID_TOKEN=$(echo "$ID_TOKEN_RESPONSE" | jq .value | tr -d '"')
echo "DEBUG: ID token length: ${#ID_TOKEN}" >&2

if [ -z "$ID_TOKEN" ] || [ "$ID_TOKEN" = "null" ]; then
    echo "ERROR: Failed to get GitHub OIDC token" >&2
    exit 1
fi

echo "DEBUG: Exchanging OIDC token for JFrog access token..." >&2
JFROG_RESPONSE=$(curl -v \
    -X POST \
    -H "Content-type: application/json" \
    ${ARTIFACTORY_URL}/access/api/v1/oidc/token \
    -d \
    "{\"grant_type\": \"urn:ietf:params:oauth:grant-type:token-exchange\", \"subject_token_type\":\"urn:ietf:params:oauth:token-type:id_token\", \"subject_token\": \"$ID_TOKEN\", \"provider_name\": \"github\"}" 2>&1)

echo "DEBUG: JFrog response: $JFROG_RESPONSE" >&2

JFROG_ACCESS_TOKEN=$(echo "$JFROG_RESPONSE" | jq .access_token -r 2>/dev/null || echo "null")

if [ -z "$JFROG_ACCESS_TOKEN" ] || [ "$JFROG_ACCESS_TOKEN" = "null" ]; then
    echo "ERROR: Failed to get JFrog access token. Response: $JFROG_RESPONSE" >&2
    exit 1
fi

echo -n $JFROG_ACCESS_TOKEN
