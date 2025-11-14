# Error Type Consolidation Plan
**Phase 3: Migration from Legacy Errors to VectorIndexError**

---

## Error Mapping Strategy

Since backward compatibility is low priority, we'll **directly replace** old error types with `VectorIndexError` at all call sites.

### 1. IVFError → VectorIndexError Mapping

```swift
// OLD: IVFError
public enum IVFError: Error {
    case invalidInput           // → .invalidParameter
    case invalidDimensions      // → .dimensionMismatch
    case invalidListID          // → .invalidRange
    case invalidFormat          // → .invalidFormat OR .unsupportedLayout
    case invalidGroup           // → .invalidParameter
    case idWidthUnsupported     // → .invalidParameter
    case capacityOverflow       // → .capacityExceeded
    case allocationFailed       // → .memoryExhausted
    case mmapRequiredForDurable // → .missingDependency
    case outOfRange             // → .invalidRange
}
```

### 2. PQError → VectorIndexError Mapping

```swift
// OLD: PQError (Int32-based C-style)
public enum PQError: Int32, Error {
    case ok = 0                 // Remove (success case)
    case invalidDim = -1        // → .invalidDimension
    case invalidK = -2          // → .invalidParameter (k parameter)
    case insufficientData = -3  // → .emptyInput
    case nullPtr = -4           // → .contractViolation (nullptr in Swift!)
    case alignment = -5         // → .contractViolation (alignment issue)
    case allocFailed = -6       // → .memoryExhausted
}
```

### 3. ResidualError → VectorIndexError Mapping

```swift
// OLD: ResidualError (Int32-based C-style)
public enum ResidualError: Int32, Error {
    case ok = 0                    // Remove (success case)
    case invalidDimension = -1     // → .invalidDimension
    case invalidCoarseID = -2      // → .invalidRange
    case invalidAlignment = -3     // → .contractViolation
    case nullPointer = -4          // → .contractViolation
    case dimensionMismatch = -5    // → .dimensionMismatch
}
```

### 4. LayoutError → VectorIndexError Mapping

```swift
// OLD: LayoutError (Associated values preserved in error context)
public enum LayoutError: Error {
    case invalidRowBlockSize(Int)        // → .invalidParameter + info("R", "\(R)")
    case invalidDimensions(n: Int, d: Int) // → .invalidDimension + info("n", "d")
    case bufferTooLarge(Int)             // → .capacityExceeded + info("size", "\(sz)")
    case invalidPQGroup(m: Int, g: Int)  // → .invalidParameter + info("m", "g")
    case invalidPQ4Bit(m: Int, g: Int)   // → .invalidParameter + info("m", "g")
}
```

### 5. IDMapError → VectorIndexError Mapping (Internal)

```swift
// OLD: IDMapError (Internal)
internal enum IDMapError: Error {
    case duplicateExternalID(UInt64)  // → .duplicateID + info("id", "\(id)")
    case tableFull                    // → .capacityExceeded
    case excessiveProbing             // → .capacityExceeded (hash table issue)
    case badBucketCount               // → .invalidParameter
    case invalidInternalID(Int64)     // → .invalidRange + info("id", "\(id)")
}
```

### 6. VIndexError → VectorIndexError Mapping (Internal)

```swift
// OLD: VIndexError (Internal mmap operations)
internal enum VIndexError: Error {
    case openFailed(errno: Int32)           // → .fileIOError + errno info
    case statFailed(errno: Int32)           // → .fileIOError + errno info
    case mmapFailed(errno: Int32)           // → .mmapError + errno info
    case badHeader                          // → .corruptedData
    case badCRC                             // → .corruptedData
    case unknownSection                     // → .invalidFormat
    case unsupportedEndianness              // → .endiannessMismatch
    case misalignedSection(expected, got)   // → .corruptedData + info
    case cannotGrowSection(SectionType)     // → .capacityExceeded
    case badListID                          // → .invalidRange
    case insufficientCapacity               // → .capacityExceeded
    case walIOFailed(errno: Int32)          // → .fileIOError + errno info
}
```

---

## Migration Order

1. ✅ **IVFError** (10 uses) - Start here, recently worked on
2. **LayoutError** (5 uses) - Public, used in LayoutTransforms
3. **PQError** (7 uses) - Public, used in PQTrain
4. **ResidualError** (6 uses) - Public, used in ResidualKernel
5. **IDMapError** (5 uses) - Internal, used in IDMap
6. **VIndexError** (12 uses) - Internal, used in VIndexMmap

---

## Implementation Strategy

For each error type:
1. **Search for all throw sites** using grep
2. **Replace inline** with appropriate `ErrorBuilder` calls
3. **Delete old error enum** after all sites migrated
4. **Run tests** to verify correctness
5. **Update affected tests** to expect `VectorIndexError`

---

**Next Step:** Migrate IVFError in IVFAppend.swift
## Status Update (2025-11-12)

- IVFError → VectorIndexError: Implemented in IVFAppend.
- VIndexError (builder/open/mmap): Replaced in `VIndexContainerBuilder` with `ErrorBuilder`.
- Remaining internal conversions (e.g., full VIndexMmap audit) are tracked in Phase 3.
<!-- moved to docs/ -->
