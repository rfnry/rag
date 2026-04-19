# HT-200 Continuous Heat Sealer — Operating Manual

## 1. Specifications

| Parameter | Value |
|-----------|-------|
| Model | HT-200 |
| Seal width | 10 mm (standard), 15 mm (optional jaw set JS-15) |
| Conveyor speed | 0-30 ft/min (variable) |
| Temperature range | 200F to 550F |
| Heating element | Nichrome wire, 500W per jaw |
| Power | 120V/15A single-phase |
| Material compatibility | PE, PP, PVC, laminated foils, coated papers |
| Maximum material thickness | 8 mil (0.008 in) per layer |
| Conveyor width | 8 inches |
| Machine dimensions | 28 x 14 x 16 inches |
| Weight | 62 lbs |

## 2. Setup and Calibration

### 2.1 Temperature Calibration

The HT-200 uses a PID temperature controller (part number TC-200) with a K-type thermocouple embedded in the upper jaw. The display shows actual jaw temperature in real-time.

Recommended starting temperatures by material:

| Material | Temperature | Speed |
|----------|-------------|-------|
| LDPE (2-4 mil) | 280F-310F | 20-25 ft/min |
| HDPE (3-6 mil) | 300F-340F | 15-20 ft/min |
| PP (2-4 mil) | 320F-360F | 15-20 ft/min |
| PVC (3-6 mil) | 250F-280F | 20-25 ft/min |
| Laminated foil | 340F-380F | 10-15 ft/min |
| Coated paper | 300F-330F | 15-20 ft/min |

These are starting points. Actual settings depend on material supplier, ambient conditions, and seal strength requirements. Always run test seals before production.

### 2.2 Jaw Pressure Adjustment

Jaw pressure is set by the two spring-loaded pressure screws on top of the upper jaw assembly. Equal pressure on both sides is critical — uneven pressure causes weak seals on one side and material distortion on the other.

To calibrate: place a strip of pressure-indicating film (Fujifilm Prescale, low-pressure range) between the jaws. Close the jaws for 3 seconds at operating temperature. The film should show uniform color across the full seal width. Adjust the pressure screws until uniformity is achieved.

Typical jaw pressure: 30-50 PSI contact pressure. Higher pressure is needed for thicker materials and laminated structures. Excessive pressure causes material thinning and premature jaw wear.

### 2.3 Conveyor Alignment

The conveyor belt must track centered on the rollers. If the belt drifts, adjust the tracking screw on the driven roller (front). Turn clockwise to move the belt right, counterclockwise to move left. Make adjustments in 1/4-turn increments and let the belt run for 30 seconds between adjustments.

Belt tension: the belt should deflect 1/4 inch when pressed firmly at the center point between rollers. Replace the belt if it shows cracks, fraying, or permanent stretching. Part number CB-200.

## 3. Operation

### 3.1 Startup Procedure

1. Turn the main power switch to ON
2. Set temperature on the PID controller to the target value
3. Wait for the "READY" indicator LED (green) — this indicates the jaw has reached setpoint. Allow an additional 2 minutes for temperature stabilization.
4. Set conveyor speed to the desired rate
5. Run 3-5 test seals and perform peel tests before production
6. If seal strength is inadequate, increase temperature by 10F or decrease speed by 2 ft/min. If seal is wrinkled or distorted, decrease temperature or increase speed.

### 3.2 Peel Test Procedure

Cut a 1-inch wide strip across the sealed area. Grip both layers and pull apart at a 180-degree angle. Acceptable seal: the material tears before the seal separates (material failure, not seal failure). If the seal peels apart cleanly, increase temperature or pressure.

For quantitative testing, use a tensile tester. Minimum seal strength for most packaging applications: 3 lbs/inch of seal width for PE, 4 lbs/inch for PP, 2.5 lbs/inch for PVC.

### 3.3 Material Changeover

When switching between material types:
1. Allow the jaws to reach the new target temperature (allow 5 minutes for cooling if reducing temperature)
2. Run the new material through the machine at slow speed (5 ft/min) for 30 seconds to purge any residue from the previous material
3. Perform test seals on the new material before production

CAUTION: PVC releases hydrochloric acid fumes when overheated. If switching from a high-temperature material (PP, laminated foil) to PVC, always reduce temperature to below 300F before running PVC. Ensure adequate ventilation when sealing PVC.

## 4. Maintenance

### 4.1 Daily Maintenance

- Clean the jaw surfaces with a soft brass brush to remove adhesive and material residue
- Inspect the PTFE (Teflon) jaw covers for wear, burn-through, or contamination. Replace if the nichrome heating wire is visible through the PTFE. Part number JC-200 (set of 2).
- Check the conveyor belt for material buildup or contamination
- Verify temperature accuracy with an external thermocouple or IR thermometer

### 4.2 Weekly Maintenance

- Inspect the silicone rubber jaw pad for compression set or hardening. Replace if the pad does not spring back when pressed. Part number JP-200.
- Clean the thermocouple tip with isopropyl alcohol (contamination causes temperature reading errors)
- Check all electrical connections for tightness (vibration can loosen terminal screws)
- Lubricate the jaw pivot bearings with one drop of food-grade machine oil

### 4.3 Quarterly Maintenance

- Replace PTFE jaw covers (JC-200) regardless of appearance — micro-cracks invisible to the eye reduce seal quality
- Inspect the heating element for hotspots (visible as discoloration on the PTFE cover). Hotspots indicate element degradation. Replace heating element HE-200 if hotspots are detected.
- Calibrate the PID controller against a reference thermocouple. If the displayed temperature differs from the reference by more than 5F, recalibrate per the TC-200 manual.

## 5. Troubleshooting

### Seals are weak or peel apart easily
1. Increase temperature by 10-20F increments
2. Decrease conveyor speed to increase dwell time
3. Increase jaw pressure
4. Check that PTFE jaw covers are not worn through
5. Verify material compatibility — not all materials are heat-sealable

### Seals are wrinkled, distorted, or burned
1. Decrease temperature by 10-20F increments
2. Increase conveyor speed
3. Decrease jaw pressure
4. Check for hotspots on heating element

### Temperature fluctuates or won't reach setpoint
1. Check thermocouple connection — loose connection causes erratic readings
2. Inspect heating element for continuity with a multimeter (expected resistance: 28-32 ohms per jaw)
3. Check PID controller settings — factory defaults: P=5, I=240, D=60
4. Verify power supply voltage — low voltage reduces heating capacity

### Conveyor belt slips or stalls
1. Check belt tension (1/4 inch deflection at center)
2. Inspect drive roller surface for contamination — clean with isopropyl alcohol
3. Check motor coupling — the set screw on the coupler may have loosened
4. Inspect the drive motor — if the motor hums but doesn't turn, the capacitor may have failed. Part number MC-120.
