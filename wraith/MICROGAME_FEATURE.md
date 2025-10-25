# 🎮 Attention Recovery Microgame Feature

## Overview
The Safety Microgame is a gamified alertness verification system designed to ensure driver safety. It detects drowsiness while driving and implements a multi-stage safety protocol.

## How It Works

### 1. **Movement Detection**
- Uses the DeviceMotion API to detect if the vehicle is moving
- Samples acceleration data and maintains a rolling average
- Threshold: 0.5 m/s² (configurable)
- Falls back to manual override toggle if DeviceMotion is unavailable

### 2. **Drowsiness Scoring**
The system calculates a drowsiness score (0-7+) based on:
- **Eyes closed duration** (+1 for >1s, +2 for >2s)
- **Gaze drift active** (+1)
- **Recent yawn** (within 5s: +1)
- **Recent head tilt** (within 5s: +1)
- **Predictive risk probability** (+1 for >0.5, +2 for >0.7)

**Alert Threshold**: Score ≥ 3 triggers the safety protocol

### 3. **Safety Protocol Stages**

#### Stage 1: Movement Alert (While Driving)
When drowsiness score ≥ 3 and vehicle is moving:
- **Persistent audio alert** (800Hz oscillating beep)
- **Full-screen warning overlay** with pulsing animation
- Message: "⚠️ DROWSINESS DETECTED - PULL OVER SAFELY NOW"
- Alert continues until vehicle stops moving

#### Stage 2: Microgame Challenge (When Stopped)
Once the vehicle stops:
- **10-second tapping challenge** launches automatically
- **10 sequential targets** appear at random positions
- Each target appears for 800ms
- Driver must tap targets quickly and accurately
- Real-time score display

#### Stage 3: Results & Actions

**Pass (8/10 or higher)**:
- ✅ Displays encouragement and safety tips:
  - Take a 15-minute break
  - Drink water or coffee
  - Do some stretches
  - Get fresh air
- Driver can continue after acknowledging

**Fail (< 8/10)**:
- ❌ Driver marked as unfit to drive
- **Company notification** sent automatically (see API integration below)
- Recommendation to arrange alternate driver or rest
- Driver can acknowledge but system logs the failure

### 4. **Company Notification API**
When a driver fails the alertness check, the system sends:

```json
{
  "timestamp": "2025-10-25T12:34:56.789Z",
  "driverId": "DRIVER_ID",
  "vehicleId": "VEHICLE_ID",
  "score": 6,
  "threshold": 8,
  "drowsinessScore": 5,
  "location": "GPS_COORDINATES"
}
```

**Integration Point**: `notifyTaxiCompany()` function in app.js
- Currently logs to console (line ~1850)
- Replace with actual API endpoint:
  ```javascript
  fetch('https://api.taxicompany.com/driver-alerts', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload)
  })
  ```

## UI Controls

In the web app sidebar under "🎮 Safety Microgame":

1. **Movement**: Shows "Moving" (red) or "Stationary" (green)
2. **Drowsiness Score**: Live score out of 3+ (threshold is 3)
3. **Manual "Stopped" Override**: For testing without actual movement

## Testing the Feature

### Method 1: Simulate Drowsiness
1. Open the app at `http://localhost:8017`
2. Start the camera
3. Check "Manual Stopped Override" (for testing)
4. Trigger drowsiness indicators:
   - Close eyes for 2+ seconds (score +2)
   - Look away to trigger gaze drift (score +1)
   - This should reach the threshold of 3
5. Uncheck "Manual Stopped" to trigger the alert
6. Check "Manual Stopped" again to launch the microgame

### Method 2: Use Real Movement
1. Open on a mobile device
2. Grant DeviceMotion permissions
3. Test in a moving vehicle (safely!)
4. Trigger drowsiness indicators
5. Pull over to stop and complete the microgame

## Safety Considerations

### When Alerts Trigger
- **Only when score ≥ 3** (moderate to high drowsiness)
- **Only when moving** (prevents false alarms when parked)
- **Persistent until stopped** (like a seatbelt alarm)

### Microgame Design
- **10 seconds** - long enough to assess alertness, short enough to be safe
- **8/10 threshold** - balances safety with realistic performance
- **Visual + motor response** - tests reaction time and coordination
- **Respectful UX** - clear, calm messaging even in fail state

### Production Recommendations
1. **Add GPS integration** for accurate location in notifications
2. **Implement driver/vehicle ID system** for tracking
3. **Set up company notification webhook** with proper authentication
4. **Add data logging** for incident review and pattern analysis
5. **Customize thresholds** based on company policy and regulations
6. **Add override codes** for emergency situations (with logging)
7. **Integrate with fleet management** systems

## Configuration

Key constants in `app.js`:

```javascript
const MOTION_THRESHOLD = 0.5;           // m/s² for "moving"
const DROWSINESS_ALERT_THRESHOLD = 3;   // Score to trigger alert
const MICROGAME_DURATION = 10000;       // 10 seconds
const MICROGAME_PASS_THRESHOLD = 8;     // Need 8/10 to pass
```

Adjust these values based on your safety requirements and testing results.

## Browser Compatibility

- **DeviceMotion API**: Requires HTTPS on mobile (http://localhost works for testing)
- **AudioContext**: All modern browsers
- **Fullscreen overlays**: All modern browsers
- **Touch/mouse events**: Universal support

## Future Enhancements

1. **Multiple game types** - variety to prevent learning the test
2. **Adaptive difficulty** - based on driver's baseline performance
3. **Historical tracking** - patterns over time
4. **Integration with wearables** - heart rate, sleep quality
5. **Voice commands** - for accessibility
6. **Multi-language support** - for international fleets
