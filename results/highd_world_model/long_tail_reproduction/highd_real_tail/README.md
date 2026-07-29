# highD real long-tail reference

The reference set is the held-out `is_evt_tail` split. It contains 328 scenes and uses the logged future solely as the evaluation target and ego replay.

| Event | Physical criterion | Scenes |
| --- | --- | ---: |
| High-risk following | minimum TTC < 3 s | 183 |
| Hard braking | minimum longitudinal acceleration < -1.5 m/s² | 17 |
| High-speed approach | maximum closing speed > 5 m/s | 162 |
| Close interaction | minimum body-clearance-adjusted gap < 8 m | 215 |
| Strong relative-speed change | one fixed slot changes relative speed by > 3 m/s over 5 s | 65 |

Events overlap by design. The fixed-slot requirement in the final row avoids falsely treating two different background vehicles as one changing interaction.
