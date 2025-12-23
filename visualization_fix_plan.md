# Visualization Controls Fix Plan

## Issue Analysis
The user reports that when selecting any chart type, the x and y parameters are not properly displayed or handled. The current `setupVisualizationControls()` function has several issues:

1. **Incomplete chart type handling**: Missing "pie" chart type
2. **Poor field management**: Not properly enabling/disabling required fields
3. **Disconnected from AJAX system**: Not coordinating with the main visualization workflow
4. **Missing parameter validation**: No proper validation of required fields per chart type

## Current State Analysis

### Working Components:
- HTML form structure with dropdowns for chart type, x_column, y_column, color_column
- AJAX visualization system that handles chart creation
- Basic chart type selection UI

### Broken Components:
- `setupVisualizationControls()` function logic
- Field requirements per chart type
- UI state management (enabling/disabling fields)
- Integration with AJAX visualization system

## Fix Plan

### 1. Update `setupVisualizationControls()` function
- Add proper handling for ALL chart types (histogram, box, pie, line, scatter, map)
- Implement correct field requirements for each chart type:
  - **Histogram**: X-axis required, Y-axis disabled, Color optional
  - **Box Plot**: Y-axis required, X-axis optional, Color optional  
  - **Pie Chart**: One column required (could be X or Y), others disabled
  - **Line Chart**: X and Y axes required, Color optional
  - **Scatter Chart**: X and Y axes required, Color optional
  - **Map**: Location column required, X/Y axes disabled, Color optional

### 2. Improve Field State Management
- Properly enable/disable fields based on chart type
- Set required attributes correctly
- Clear values for disabled fields
- Update field labels dynamically (e.g., "Color Column" vs "Location Column")

### 3. Enhance Integration with AJAX System
- Ensure the function coordinates with the existing AJAX visualization workflow
- Trigger visualization creation when fields change
- Handle real-time updates and validation

### 4. Add Better UX
- Visual feedback for required vs optional fields
- Clear indication of what each field represents
- Better error handling and user guidance

## Implementation Steps

1. **Replace the current `setupVisualizationControls()` function** with a comprehensive version
2. **Test each chart type** to ensure proper field handling
3. **Verify AJAX integration** works correctly
4. **Add visual feedback** for better user experience

## Expected Outcome
After the fix, users should be able to:
- Select any chart type and see appropriate parameter fields
- Have clear indication of which fields are required vs optional
- Get immediate visual feedback when fields are enabled/disabled
- Create charts successfully with proper parameter validation
