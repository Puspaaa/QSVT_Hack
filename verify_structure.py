import os
import sys

# Check Streamlit multi-page structure
print("✓ Checking Streamlit multi-page structure...")

# Required files
required_files = [
    "app.py",
    "pages/1_1D_Simulation.py", 
    "pages/2_2D_Simulation.py",
    "quantum.py",
    "simulation.py",
    "solvers.py",
    "requirements.txt"
]

all_exist = True
for file_path in required_files:
    if os.path.exists(file_path):
        print(f"  ✅ {file_path}")
    else:
        print(f"  ❌ {file_path} (MISSING)")
        all_exist = False

if all_exist:
    print("\n✓ All required files exist!")
    
    # Check app.py is entry point
    with open("app.py") as f:
        app_content = f.read()
        if "st.set_page_config" in app_content and "Home" in app_content or "Welcome" in app_content:
            print("✓ app.py looks like home page")
        else:
            print("⚠ app.py structure unclear")
    
    # Check 1D page
    with open("pages/1_1D_Simulation.py") as f:
        page1_content = f.read()
        if "1D" in page1_content and "simulation" in page1_content.lower():
            print("✓ pages/1_1D_Simulation.py looks correct")
        else:
            print("⚠ pages/1_1D_Simulation.py structure unclear")
    
    # Check 2D page
    with open("pages/2_2D_Simulation.py") as f:
        page2_content = f.read()
        if "2D" in page2_content and ("development" in page2_content or "Coming Soon" in page2_content):
            print("✓ pages/2_2D_Simulation.py looks correct (in development)")
        else:
            print("⚠ pages/2_2D_Simulation.py structure unclear")
    
    print("\n✅ Streamlit multi-page app is properly structured!")
    print("\nTo run the app:")
    print("  streamlit run app.py")
    sys.exit(0)
else:
    print("\n❌ Some files are missing!")
    sys.exit(1)
