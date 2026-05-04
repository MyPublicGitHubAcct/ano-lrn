# Precedence Effect

## `haas`

The Haas (precedence) effect exploits the ear's directional fusion mechanism. When the same signal arrives at two ears within about 40 ms of each other, the brain fuses the two arrivals into a single perceived source whose location is biased toward the first arrival:

```text
L[n] = x[n]
R[n] = x[n − D]
```

At delay values D < ~1750 samples (< 40 ms at 44100 Hz) the right channel is heard as part of the same source, creating a sense of width without a distinct echo. Beyond ~40 ms the delay becomes a perceptible echo.
