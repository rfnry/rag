# XR-500 Industrial Air Compressor — Service Manual

## 1. Specifications

| Parameter | Value |
|-----------|-------|
| Model | XR-500 |
| Type | Two-stage reciprocating |
| Motor | 5 HP, 230V/460V 3-phase |
| RPM | 3,450 |
| Maximum discharge pressure | 175 PSI |
| Tank capacity | 80 gallons |
| CFM at 90 PSI | 18.5 |
| CFM at 175 PSI | 15.2 |
| Oil type | SAE 30 non-detergent compressor oil |
| Oil capacity | 1.5 quarts |
| Weight | 485 lbs |
| Noise level | 82 dB at 3 ft |
| Operating temperature range | 40F to 110F ambient |

## 2. Installation

### 2.1 Foundation Requirements

The XR-500 must be mounted on a level concrete floor capable of supporting 600 lbs static load plus vibration forces. Use the four 1/2-inch mounting bolts provided. Install rubber vibration isolation pads (part number VP-100) between the base and floor to reduce transmitted vibration.

Minimum clearance: 24 inches on all sides for ventilation, 36 inches in front of the control panel for service access.

### 2.2 Electrical Connection

The XR-500 requires a dedicated circuit. For 230V operation: 30A breaker, 10 AWG wire, NEMA 6-30R outlet. For 460V operation: 15A breaker, 14 AWG wire, NEMA 15-30R outlet. All wiring must comply with NEC Article 430.

The motor starter is factory-configured for 230V. To convert to 460V: open the junction box on the motor housing, move leads T1/T4 to the HIGH position, move leads T2/T5 to the HIGH position, reconnect T3/T6 per the wiring diagram on the inside of the junction box cover.

WARNING: Disconnect all power before opening the junction box. Verify zero voltage with a multimeter before touching any leads. Failure to follow lockout/tagout procedures can result in death or serious injury.

### 2.3 Initial Oil Fill

Remove the oil fill cap on the crankcase (left side, yellow cap). Fill with 1.5 quarts of SAE 30 non-detergent compressor oil. Check the sight glass — oil level must be between the MIN and MAX marks. Do not use automotive motor oil, synthetic oil, or multi-viscosity oil. These will cause premature reed valve failure.

## 3. Operation

### 3.1 Startup Procedure

1. Verify oil level is between MIN and MAX on sight glass
2. Open the tank drain valve for 5 seconds to expel condensation
3. Close tank drain valve
4. Verify all air outlets are closed or connected to downstream equipment
5. Set the pressure switch to desired cut-out pressure (factory default: 175 PSI)
6. Turn the ON/OFF switch to ON
7. The compressor will run until tank pressure reaches cut-out setting, then stop automatically
8. The compressor will restart when tank pressure drops to cut-in setting (factory default: 145 PSI)

### 3.2 Thermal Overload Protection

The XR-500 is equipped with an automatic thermal overload protector on the motor. If the motor temperature exceeds 285F, the thermal overload will trip and the compressor will stop. The thermal overload resets automatically when the motor cools.

If the thermal overload trips repeatedly:
- Check ambient temperature (must be below 110F)
- Check ventilation clearance (24 inches minimum on all sides)
- Check that voltage is within 10% of nameplate rating
- Inspect the cooling fan for obstructions
- Check the intake filter for blockage (dirty intake filter is the #1 cause of thermal trips)

### 3.3 Pressure Switch Adjustment

The pressure switch (part number PS-200) controls cut-in and cut-out pressures. Factory defaults: cut-in 145 PSI, cut-out 175 PSI (30 PSI differential).

To adjust: remove the pressure switch cover. The large adjustment screw sets the cut-out pressure. Clockwise increases pressure, counterclockwise decreases. The small adjustment screw sets the differential (difference between cut-in and cut-out). Do not set cut-out above 200 PSI or differential below 20 PSI.

## 4. Maintenance Schedule

| Interval | Task | Part Number |
|----------|------|-------------|
| Daily | Check oil level, drain tank condensation | — |
| Weekly | Inspect intake filter, check for air leaks | — |
| 250 hours | Replace intake filter element | IF-300 |
| 500 hours | Change compressor oil | — |
| 1,000 hours | Inspect reed valves | RV-2201 (replacement kit) |
| 1,000 hours | Inspect belt tension and condition | BT-150 (belt) |
| 2,000 hours | Replace pressure switch diaphragm | PS-200D |
| 5,000 hours | Full rebuild — pistons, rings, gaskets | RK-500 (rebuild kit) |

## 5. Troubleshooting

### Compressor will not start
1. Check power supply — verify voltage at disconnect
2. Check ON/OFF switch — replace if damaged
3. Check thermal overload — press reset button on motor housing
4. Check pressure switch — if tank is already at cut-out pressure, the compressor will not start. Drain some air to reduce pressure below cut-in.
5. Check motor capacitor (single-phase models only) — a failed capacitor prevents starting. Part number MC-230.

### Compressor runs but does not build pressure
1. Check for air leaks at all fittings and connections — use soapy water spray
2. Inspect reed valves in cylinder head for carbon buildup or damage. Remove the four hex bolts on the cylinder head cover. Reed valves should lay flat against the valve seat with no visible gaps. Replace if warped, cracked, or carboned. Part number RV-2201.
3. Check the unloader valve — if stuck open, air escapes during compression. Part number UV-100.
4. Check the intake filter — a severely clogged filter reduces airflow below usable threshold.

### Excessive oil consumption
1. Check for oil leaks at crankcase gasket, sight glass, and oil fill cap
2. Inspect piston rings for wear — if oil appears in the discharged air, rings are worn. Rebuild kit RK-500.
3. Verify you are using the correct oil (SAE 30 non-detergent only)

### Excessive noise or vibration
1. Check mounting bolts — torque to 45 ft-lbs
2. Inspect vibration isolation pads VP-100 for deterioration
3. Check belt tension — belt should deflect 1/2 inch with 10 lbs of force at the midpoint
4. Inspect flywheel and pulley for cracks
5. Check connecting rod bearings — knocking noise indicates bearing wear. Rebuild required.
