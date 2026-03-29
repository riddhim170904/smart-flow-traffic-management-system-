// Pin mapping for 4 lanes, 3 colors each
const int redPins[4]    = {2, 5, 8, 11};
const int yellowPins[4] = {3, 6, 9, 12};
const int greenPins[4]  = {4, 7, 10, 13};

void setup() {
  for (int i = 0; i < 4; i++) {
    pinMode(redPins[i], OUTPUT);
    pinMode(yellowPins[i], OUTPUT);
    pinMode(greenPins[i], OUTPUT);
    // Start with all red
    digitalWrite(redPins[i], HIGH);
    digitalWrite(yellowPins[i], LOW);
    digitalWrite(greenPins[i], LOW);
  }
  Serial.begin(9600);
}

void setLane(int lane, int red, int yellow, int green) {
  digitalWrite(redPins[lane], red);
  digitalWrite(yellowPins[lane], yellow);
  digitalWrite(greenPins[lane], green);
}

void setAllRed() {
  for (int i = 0; i < 4; i++) {
    setLane(i, HIGH, LOW, LOW);
  }
}

void loop() {
  if (Serial.available()) {
    String cmd = Serial.readStringUntil('\n');
    cmd.trim();
    if (cmd.startsWith("L")) {
      int lane = cmd.substring(1, 2).toInt() - 1; // 0-based
      if (lane >= 0 && lane < 4) {
        if (cmd.endsWith("RED")) {
          setLane(lane, HIGH, LOW, LOW);
        } else if (cmd.endsWith("YELLOW")) {
          setLane(lane, LOW, HIGH, LOW);
        } else if (cmd.endsWith("GREEN")) {
          setLane(lane, LOW, LOW, HIGH);
        }
      }
    }
  }
}