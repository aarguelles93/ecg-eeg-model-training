
"""
Analysis of the EEG generation script to check for any normalization steps
"""

import re

def analyze_eeg_generation_script():
    """Analyze the generate_eeg_csv_from_bdf.py script for normalization"""
    
    script_path = "generate_eeg_csv_from_bdf.py"
    
    print("🔍 ANALYZING EEG GENERATION SCRIPT FOR NORMALIZATION")
    print("=" * 60)
    
    try:
        with open(script_path, 'r') as f:
            content = f.read()
        
        print(f"📁 Script: {script_path}")
        print(f"📄 Lines: {len(content.splitlines())}")
        
        # Check for normalization keywords
        normalization_keywords = [
            'normalize', 'normalization', 'standardize', 'standardization',
            'scale', 'scaling', 'zscore', 'z-score', 'minmax', 'min-max',
            'StandardScaler', 'MinMaxScaler', 'RobustScaler',
            'mean', 'std', 'subtract', 'divide'
        ]
        
        print(f"\n🔍 SEARCHING FOR NORMALIZATION KEYWORDS:")
        found_keywords = []
        
        lines = content.splitlines()
        for i, line in enumerate(lines, 1):
            line_lower = line.lower()
            for keyword in normalization_keywords:
                if keyword.lower() in line_lower:
                    found_keywords.append((keyword, i, line.strip()))
        
        if found_keywords:
            print(f"   ⚠️ Found {len(found_keywords)} potential normalization references:")
            for keyword, line_num, line_text in found_keywords:
                print(f"     Line {line_num}: '{keyword}' in '{line_text}'")
        else:
            print(f"   ✅ No explicit normalization keywords found")
        
        # Check for mathematical operations that could be normalization
        print(f"\n🧮 CHECKING FOR MATHEMATICAL OPERATIONS:")
        
        math_patterns = [
            (r'[-+*/]\s*np\.mean', 'Mean operations'),
            (r'[-+*/]\s*np\.std', 'Standard deviation operations'),
            (r'[-+*/]\s*\.mean\(\)', 'Mean method calls'),
            (r'[-+*/]\s*\.std\(\)', 'Std method calls'),
            (r'/\s*\d+\.?\d*', 'Division by constants'),
            (r'-\s*\d+\.?\d*', 'Subtraction of constants'),
        ]
        
        math_found = []
        for pattern, description in math_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                # Find line number
                line_num = content[:match.start()].count('\n') + 1
                line_text = lines[line_num - 1].strip()
                math_found.append((description, line_num, line_text, match.group()))
        
        if math_found:
            print(f"   ⚠️ Found {len(math_found)} mathematical operations:")
            for desc, line_num, line_text, match_text in math_found:
                print(f"     Line {line_num}: {desc} - '{match_text}' in '{line_text}'")
        else:
            print(f"   ✅ No suspicious mathematical operations found")
        
        # Check for data transformation steps
        print(f"\n🔄 CHECKING DATA TRANSFORMATION STEPS:")
        
        transformation_patterns = [
            (r'\.flatten\(\)', 'Data flattening (OK)'),
            (r'\.reshape\(', 'Data reshaping (OK)'),
            (r'\.transpose\(', 'Data transposition (OK)'),
            (r'\.copy\(\)', 'Data copying (OK)'),
            (r'\.astype\(', 'Data type conversion (OK)'),
            (r'np\.clip\(', 'Data clipping (potential normalization)'),
            (r'np\.log\(', 'Logarithmic transformation'),
            (r'np\.exp\(', 'Exponential transformation'),
            (r'np\.power\(', 'Power transformation'),
        ]
        
        transformations_found = []
        for pattern, description in transformation_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                line_num = content[:match.start()].count('\n') + 1
                line_text = lines[line_num - 1].strip()
                transformations_found.append((description, line_num, line_text))
        
        if transformations_found:
            print(f"   Found {len(transformations_found)} data transformations:")
            for desc, line_num, line_text in transformations_found:
                print(f"     Line {line_num}: {desc} - '{line_text}'")
        else:
            print(f"   ✅ No data transformations found")
        
        # Check MNE-specific operations
        print(f"\n🧠 CHECKING MNE-SPECIFIC OPERATIONS:")
        
        mne_operations = [
            'raw.filter(',
            'raw.resample(',
            'raw.pick_types(',
            'raw.pick_channels(',
            'raw.get_data(',
            'raw.apply_function(',
            'raw.apply_proj(',
            'mne.preprocessing.',
        ]
        
        mne_found = []
        for operation in mne_operations:
            if operation.lower() in content.lower():
                # Find all occurrences
                for i, line in enumerate(lines, 1):
                    if operation.lower() in line.lower():
                        mne_found.append((operation, i, line.strip()))
        
        if mne_found:
            print(f"   Found {len(mne_found)} MNE operations:")
            for operation, line_num, line_text in mne_found:
                print(f"     Line {line_num}: {operation} - '{line_text}'")
        else:
            print(f"   ✅ No MNE operations found")
        
        # Check for imports that might do normalization
        print(f"\n📦 CHECKING IMPORTS:")
        
        import_lines = [line for line in lines if 'import' in line.lower()]
        suspicious_imports = []
        
        normalization_modules = ['sklearn.preprocessing', 'scipy.stats', 'statsmodels']
        
        for line in import_lines:
            for module in normalization_modules:
                if module in line.lower():
                    suspicious_imports.append(line.strip())
        
        if import_lines:
            print(f"   All imports:")
            for imp in import_lines:
                print(f"     {imp.strip()}")
            
            if suspicious_imports:
                print(f"   ⚠️ Suspicious imports (normalization-related):")
                for imp in suspicious_imports:
                    print(f"     {imp}")
            else:
                print(f"   ✅ No normalization-related imports")
        
        # Final assessment
        print(f"\n🎯 ASSESSMENT:")
        
        issues_found = len(found_keywords) + len([m for m in math_found if 'mean' in m[0].lower() or 'std' in m[0].lower()])
        
        if issues_found == 0:
            print(f"   ✅ SCRIPT APPEARS CLEAN - No explicit normalization found")
            print(f"   📝 The script only extracts raw EEG data and flattens it")
            print(f"   🔍 Normalization is likely happening elsewhere in your pipeline")
        else:
            print(f"   ⚠️ POTENTIAL NORMALIZATION DETECTED - {issues_found} suspicious operations")
            print(f"   🔍 Review the flagged lines above")
        
        # Specific analysis of the key data processing line
        print(f"\n🔍 KEY DATA PROCESSING ANALYSIS:")
        
        # Look for the main data extraction line
        data_extraction_lines = []
        for i, line in enumerate(lines, 1):
            if 'get_data' in line or 'segment' in line.lower() or 'flatten' in line:
                data_extraction_lines.append((i, line.strip()))
        
        if data_extraction_lines:
            print(f"   Data extraction/processing lines:")
            for line_num, line_text in data_extraction_lines:
                print(f"     Line {line_num}: {line_text}")
        
        return {
            'normalization_keywords_found': len(found_keywords),
            'math_operations_found': len(math_found),
            'likely_clean': issues_found == 0,
            'suspicious_lines': found_keywords + [(m[0], m[1], m[2]) for m in math_found]
        }
        
    except FileNotFoundError:
        print(f"❌ Script not found: {script_path}")
        print(f"   Make sure the script is in the current directory")
        return None
    except Exception as e:
        print(f"❌ Error analyzing script: {e}")
        return None

if __name__ == "__main__":
    result = analyze_eeg_generation_script()
    
    print(f"\n" + "="*60)
    if result and result['likely_clean']:
        print(f"✅ EEG GENERATION SCRIPT IS LIKELY CLEAN")
        print(f"🔍 Look for normalization in:")
        print(f"   1. ECG dataset preprocessing")
        print(f"   2. Data loading functions in your main pipeline")
        print(f"   3. Previous preprocessing steps")
    else:
        print(f"⚠️ POTENTIAL NORMALIZATION FOUND IN EEG SCRIPT")
        print(f"📝 Review the flagged lines above")
    print(f"="*60)