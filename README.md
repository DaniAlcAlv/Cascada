# 🛠 CaSCaDA: Calibration and Spot Check and Dashboard Assitant

A toolkit to load, validate, visualize, create, and update **Water Valve Calibrations** and **Spotchecks** used in VR foraging rigs.

## 🎯 Purpose
- Display rig schemas, calibrations, and spotchecks  
- Validate calibrations (regression, consistency, aging, bounds)  
- Create and save **manual calibrations entries**  
- Create and save **spotcheck entries**  
- Update **rig schema JSON** from any WaterCal file  


## 📁 Project Structure
src/  
│  
├─ CaSCaDa.py                 # Streamlit entrypoint  
│  
├─ config.py                  # Default paths and constants  
│  
├── models
│   ├── watercal_model.py     # Calibration models and regression logic  
│   ├── watercal_dataset.py   # Dataset loader and record indexing  
│   └── spotcheck_model.py    # Spotcheck models and dataframe loader  
│  
├── services/                  
│   ├── cache.py              # Streamlit caching helpers  
│   ├── dataset_loader.py     # Wrappers for loading datasets  
│   └── plotting.py           # Matplotlib → PNG conversion  
│   
├─ ui/      
│   ├─ filters.py             # Sidebar filters  
│   └─ record_block.py        # Shared calibration display UI  
│  
├─ pages/                     # Streamlit pages 
│   ├─ rig_dashboard.py  
│   ├─ watercal_dashboard.py  
│   ├─ spotcheck_dashboard.py  
│   ├─ new_spotcheck.py  
│   └─ manual_calibration.py  
│  
├─ PyGUI.py                   # Terminal interface  
│  
└─ extras/                    # Other scripts

## 📦 Supported WaterCal Data Layouts
### Automatic Calibration
*/water_calibration.json  (Contains the measurements and regression)  
*/behavior/logs/rig_input.json  (Contains rig info) 

### Manual Calibration
*/water_calibration.json  (Contains the measurements and regression)   
*/rig_info.json  (Contains rig info)   

### Rig Schema  
*/[computer_name]/**.json (Contains both the rig info and the water calibration data):  

## ✔ Validation
- JSON load & structure  
- repeat_count ≥ 21  
- at least 2 measurement rows  
- interval_average consistency  
- positive and finite values  
- regression recomputation and thresholds  
- calibration age checks  

## 🖥 WebGUI (Streamlit)

### **Current Rig Dashboard**
- View calibrations inside rig schemas  
- Show errors/warnings  
- Plot regression & intervals  
- Update a rig’s calibration block in-place preserving all other keys and creates a .bak backup  

### **Water Calibration Dashboard**
- Historical standalone calibrations  
- Diagnostics + plots  

### **Spotcheck Dashboard**
- All rigs: Displays a table with data from the last calibration and spotchecks
- Single rig: KPIs, Ratio-to-target plot, Error % timeline  plot

### **New Spotcheck**
- Allows the user to select a rig and schema to auto‑compute repeat count & valve time, and then enter delivered mass  
- OK/Strike/Fail shown visually  
- Save spotcheck JSON  

### **Manual Calibration**
- Enter new measurements 
- Shows diagnostics + regression plot  
- Save the results in calibration folder
