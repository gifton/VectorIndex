#!/bin/bash

test_filter() {
    local filter="$1"
    local name="$2"
    echo "Testing: $name"
    result=$(swift test --filter "$filter" 2>&1 | grep -E "Executed [0-9]+ test|warning: No matching")
    if echo "$result" | grep -q "warning: No matching"; then
        echo "  ❌ No tests found"
    else
        count=$(echo "$result" | grep -oE "Executed [0-9]+" | head -1 | awk '{print $2}')
        echo "  ✅ Found $count tests"
    fi
    echo
}

# Test various filter patterns
test_filter '^(FlatIndexTests|FlatIndexEdgeCasesTests)\.' "Flat (without prefix + with dot)"
test_filter '^VectorIndexTests\.(FlatIndexTests|FlatIndexEdgeCasesTests)\.' "Flat (with prefix + with dot)"
test_filter '^VectorIndexTests\.(FlatIndexTests|FlatIndexEdgeCasesTests)' "Flat (with prefix + no dot)"

test_filter '^(KMeansMiniBatchTests|KMeansPPSeedingTests)\.' "KMeans (without prefix)"
test_filter '^VectorIndexTests\.(KMeansMiniBatchTests|KMeansPPSeedingTests)' "KMeans (with prefix)"

test_filter '^PersistenceTests\.' "Persistence (without prefix)"
test_filter '^VectorIndexTests\.PersistenceTests' "Persistence (with prefix)"

