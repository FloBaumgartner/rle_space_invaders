# Bericht

## Baseline

## initialer Ansatz

## Experimentelles Setup

## Erweiterungen



### IMPALA-CNN-Backbone
Ersetzt die kleine CNN durch ein dreistufiges IMPALA-ResNet.
hier keine layer initialisierung, da torch mit mit kaiming he (relu) initialisiert.
Das IMPALA-Backbone hat Residualverbindungen und Max-Pooling;
dadurch verteilt sich der Gradient ohnehin besser,
sodass die Vorteile einer expliziten Orthogonal init hier geringer ausfallen.
#### Hypothese
Dreistufige Residual-ähnliche Convolutional-Blöcke mit Pooling sollten reichhaltigere und hierarchischere Features
extrahieren als das schmalere Classic-CNN. Ich erwartete dadurch schnellere Konvergenz und eine robustere Repräsentation,
die höhere mittlere Return-Werte in SpaceInvaders ermöglicht.
#### Parameter / änderungen
#### evaluation der Ergebnisse
#### Analyse

### Recurrent PPO (LSTM)
Diese Methode hat nicht nur ein Architektur-Parameter-Tweak (wie IMPALA-CNN), sondern auch anderer Rechen-Graph (hidden state, sequence batching, …).
#### Hypothese
Ein R-PPO-Agent kann aus der Historie eine latent-State-Schätzung bilden. Durch Hinzufügen eines LSTM-Layers hinter dem
Feature-Extractor sollte der Agent ein Gedächtnis für zeitlich entfernte Abhängigkeiten geben.
Ich erwarte signifikanten Reward-Zuwachs in Umgebungen, in denen die letzte Observation allein nicht ausreicht, 
um optimale Aktionen zu wählen. Gerade in Enviroments mit teilweiser Beobachtbarkeit (z. B. Bewegungsmuster der Gegner)
könnte das Langzeitgedächtnis zu stabileren, strategischeren Entscheidungen und besseren Return-Werten führen.

#### Parameter / änderungen
#### evaluation der Ergebnisse
#### Analyse

### Random Network Distillation (RND)
Fügt einen RND-Bonus als additive Reward-Komponente ein. Es sollte Sparse-Reward-Probleme mindern und
erhöht Policy-Entdeckung neuer States.
#### Hypothese
Die intrinsische Belohnung aus der Prädiktor-Fehler-Signalbibliothek zielt darauf ab, Neugier zu fördern und selten besuchte
Zustände häufiger zu erkunden. Ich erwartete, dass der Agent so aus seiner Exploitation-Schleife ausbricht,
neue Taktiken entdeckt und insgesamt höhere extrinsische Return-Werte erzielt, insbesondere in Phasen mit spärlichem 
Umgebungs-Reward.
#### Parameter / änderungen
#### evaluation der Ergebnisse
#### Analyse

### Hyperparameter-Optimierung mit Optuna
Bayesian Search auf LR, n_steps, clip-range, λ u. a.
Automatisches Auffinden gut abgestimmter HPs liefert oft schnellere Konvergenz und bessere Ergebnisse als manuelles Tuning.

#### Hypothese
Durch die systematische Suche (Bayesian/TPE-Sampler) erwartete ich, wesentlich besser abgestimmte Lernraten, 
Mini-Batch-Größen und Clip-Koeffizienten zu finden als durch manuelles Raten.
Das sollte die Stichprobeneffizienz steigern und das Risiko suboptimaler Konfigurationen minimieren. Insbesondere bei "nur" 5 Mio. Frames Trainingsbudget.
#### Parameter / änderungen
#### evaluation der Ergebnisse
#### Analyse

## FInaler Vergleich

## Fazit 

## Ausblick