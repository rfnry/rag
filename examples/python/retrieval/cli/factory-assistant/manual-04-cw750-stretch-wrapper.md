# CW-750 Automatic Stretch Wrapper — Service Manual

## 1. Specifications

| Parameter | Value |
|-----------|-------|
| Model | CW-750 |
| Type | Turntable, fully automatic |
| Turntable diameter | 65 inches |
| Maximum load dimensions | 52 x 52 x 72 inches |
| Maximum load weight | 4,000 lbs |
| Turntable speed | 5-15 RPM (variable) |
| Film carriage travel | 72 inches vertical |
| Film type | 80 gauge cast stretch film (standard), 60-120 gauge range |
| Film roll dimensions | 20-inch width, 3-inch core |
| Pre-stretch ratio | 200% (fixed gear), 250% optional |
| Power | 230V/20A single-phase |
| Air supply | Not required (all-electric) |
| PLC | Allen-Bradley Micro850 |
| Weight | 2,800 lbs |

## 2. Film Threading

### 2.1 Loading a New Film Roll

1. Open the film carriage access door
2. Remove the spent core from the film spindle
3. Place the new roll on the spindle with the film unwinding from the bottom (clockwise when viewed from the left side)
4. Tighten the spindle lock knob
5. Thread the film through the pre-stretch rollers: over the first roller (idler), under the second roller (driven), over the third roller (idler)
6. Pull approximately 3 feet of film and attach to the load or the film clamp on the turntable
7. Close the access door

### 2.2 Pre-Stretch System

The CW-750 uses a mechanical pre-stretch system with a fixed gear ratio. The driven roller rotates faster than the film unwinds, stretching the film before it contacts the load. Standard gear ratio provides 200% pre-stretch (1 foot of film from the roll becomes 3 feet of stretched film).

Benefits of pre-stretch:
- Reduces film consumption by 50-65%
- Increases holding force (stretched film has higher elastic recovery)
- Reduces film cost per pallet by approximately 60%

The 250% pre-stretch gear set (part number PG-250) can be installed for thinner films (60-80 gauge). Do not use 250% pre-stretch with films heavier than 80 gauge — the increased tension can tear the film or damage the pre-stretch rollers.

## 3. Wrap Cycle Programming

### 3.1 Cycle Parameters

The wrap cycle is defined by the following parameters, accessible on the HMI:

| Parameter | Range | Default | Description |
|-----------|-------|---------|-------------|
| Bottom wraps | 1-9 | 3 | Extra wraps at the base for stability |
| Up speed | 1-99% | 40% | Film carriage speed going up |
| Down speed | 1-99% | 60% | Film carriage speed going down |
| Top wraps | 1-9 | 2 | Extra wraps at the top |
| Turntable speed | 5-15 RPM | 10 | Rotation speed |
| Up/down cycles | 1-5 | 2 | Number of up-down passes |
| Film tension | 1-99% | 50% | Brake force on the film delivery |
| Top delay | 0-10 sec | 2 | Pause at top before reversing |

### 3.2 Creating a Wrap Program

The CW-750 stores up to 10 wrap programs. Access MENU > PROGRAMS > EDIT.

**Light loads** (under 500 lbs, stable cases): 2 bottom wraps, 1 up/down cycle, 30% film tension, turntable speed 12 RPM.

**Standard loads** (500-2000 lbs, uniform cases): 3 bottom wraps, 2 up/down cycles, 50% film tension, turntable speed 10 RPM.

**Heavy loads** (2000-4000 lbs, irregular shapes): 4 bottom wraps, 3 up/down cycles, 70% film tension, turntable speed 8 RPM. Consider applying a rope pattern at the top (MENU > PROGRAMS > ROPE MODE) for additional vertical containment.

### 3.3 Film Break Detection

The CW-750 includes an ultrasonic film break sensor (part number FBS-100) mounted on the film carriage. If the sensor detects a break:
1. The turntable stops immediately
2. The film carriage stops
3. The HMI displays FILM BREAK — RETHREAD AND PRESS START
4. Rethread the film per section 2.1, attach to the load, and press START to resume the cycle from the current position

## 4. Maintenance

### 4.1 Daily

- Inspect the turntable surface for debris, spilled product, or film scraps. Debris can cause load instability.
- Check the film roll — replace when the roll diameter is less than 4 inches (prevents film tension inconsistencies at the core).
- Verify the film clamp (part number FC-100) operates smoothly — the clamp grips the film tail at the end of the cycle. Clean adhesive residue from the clamp jaw with isopropyl alcohol.

### 4.2 Weekly

- Inspect the pre-stretch rollers for film buildup, adhesive residue, or scoring. Clean with isopropyl alcohol and a non-abrasive pad. Scored rollers cause film tracking problems and must be replaced: primary roller PR-100, secondary roller PR-200.
- Check the film carriage chain tension — the chain should have 1/4 inch of play at the midpoint. Adjust with the tensioner bolt at the top of the mast.
- Lubricate the film carriage guide rails with light machine oil (2-3 drops per rail).
- Check the turntable drive belt for wear — part number TB-750. Replace if cracked or glazed.

### 4.3 Semi-Annual

- Replace the film clamp pad (FC-100P) — the rubber pad hardens over time and loses grip
- Inspect the turntable bearing for noise or roughness — spin the empty turntable by hand and listen. Grinding noise indicates bearing failure. Part number TBR-750.
- Check the film carriage limit switches (top and bottom) for proper operation — manually trigger each switch and verify the HMI diagnostic page reflects the state change
- Inspect all wiring in the control panel for loose connections, discoloration, or heat damage

## 5. Troubleshooting

### Film breaks during wrapping
1. Reduce film tension (decrease by 10% increments)
2. Check for sharp edges on the load — apply corner protectors if needed
3. Inspect pre-stretch rollers for scoring or debris (primary cause of film breaks)
4. Verify the film gauge matches the pre-stretch ratio — do not use 60 gauge film with 200% pre-stretch on heavy loads
5. Check that the film roll is correctly oriented (film unwinds from the bottom)

### Load is unstable after wrapping
1. Increase bottom wraps (add 1-2 extra)
2. Increase film tension (increase by 10% increments)
3. Reduce turntable speed — faster rotation can cause the film to "neck down" and lose width coverage
4. Add an up/down cycle
5. Verify the load is centered on the turntable — an off-center load creates uneven film distribution

### Turntable won't rotate
1. Check that the load weight is within the 4,000 lb limit
2. Verify the turntable is not mechanically blocked by debris or a misaligned pallet
3. Check the turntable drive belt TB-750 for breakage
4. Inspect the turntable motor for overload trip — reset button is on the motor junction box
5. Check the proximity sensor (PS-TT) that detects turntable home position — the turntable must pass the home sensor to start a cycle

### Film carriage won't move
1. Check the carriage chain tension — too tight prevents movement, too loose causes skipping
2. Inspect the top and bottom limit switches — a failed switch prevents travel in one direction
3. Check the carriage motor — listen for the motor energizing when the cycle starts. No sound indicates a motor or contactor issue.
4. Verify the film carriage is not mechanically jammed — check for film wrapped around the guide rails
