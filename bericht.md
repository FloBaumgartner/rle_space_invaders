# Bericht

## Baseline
Als Baseline dienen zwei sehr unterschiedliche Ansätze, die keine echten Lernkomponenten enthalten, aber wertvolle Orientierungspunkte liefern:

Der RandomAgent wählt bei jedem Schritt zufällig eine Aktion aus dem diskreten Aktionsraum von Space Invaders. Diese „Dummybaseline“ zeigt, welchen Durchschnitts-Score man ohne jegliche Strategie erzielt und dient dazu, den minimal erwartbaren Reward zu definieren. Ich erwarte über 100 Evaluationsepisoden nur sehr niedrige Punktzahlen und sehr kurze Episodenlängen, da er keine zielgerichteten Schüsse abgibt und keine Positionierungsvorteile nutzt.

Demgegenüber steht der HeuristicAgent, ein einfacher regelbasierter Agent, der stets gleichzeitig schießt (FIRE) und sich abwechselnd für jeweils eine feste Anzahl von Frames nach links oder rechts bewegt. Technisch wird hierzu bei jedem Schritt die Aktion RIGHTFIRE oder LEFTFIRE ausgewählt und nach einer vordefinierten Anzahl von Schritten (bounce_steps) die Bewegungsrichtung invertiert. Diese Heuristik sorgt zumindest dafür, dass der Agent kontinuierlich feuert und das Raumschiff in Bewegung bleibt, wodurch er deutlich höhere Punktzahlen erreicht als der RandomAgent, aber natürlich immer noch weit von lernenden Algorithmen entfernt ist.

Beide Baselines werden über jeweils 100 Episoden evaluiert, wobei neben dem kumulierten Reward auch die Episodenlänge und die Laufzeit gemessen werden. So erhält man nicht nur einen Vergleich auf Basis des Scores, sondern auch Einsichten in Stabilität und Effizienz – wichtige Referenzwerte für alle folgenden, lernbasierten Agenten.

Ich habe bewusst 2 Baselines gewählt, um ein breites Spektrum an Vergleichswerten zu erhalten: Der RandomAgent zeigt die untere Grenze des Möglichen, während der HeuristicAgent eine einfache, aber effektive Strategie demonstriert. Beide liefern wertvolle Orientierungspunkte für die Bewertung der späteren lernenden Agenten.

### Ergebnisse des HeuristicAgent
<img src="plots/base_heur_ep_length_bp.png" alt="Episode length plot" height="120" /><img src="plots/base_heur_ep_return_bp.png" alt="Episode return plot" height="120" /><img src="plots/base_heur_ep_time_bp.png" alt="Episode time plot" height="120" />

Der heuristische Agent erzielt einen durchschnittlichen `Episodic return` von ca. `265` und eine Standartabweichung von  ca. `105`.
### Ergebnisse des RandomAgent
<img src="plots/base_rand_ep_length_bp.png" alt="Episode length plot" height="120" /><img src="plots/base_rand_ep_return_bp.png" alt="Episode return plot" height="120" /><img src="plots/base_rand_ep_time_bp.png" alt="Episode time plot" height="120" />

Der random Agent erzielt einen durchschnittlichen `Episodic return` von ca. `146` und eine Standartabweichung von  ca. `106`.

### Zusammenfassung der Baseline-Ergebnisse
Der heuristische Agent liefert mit einer Mittelrendite von ≈265 gegenüber ≈146 des Zufallagents schon einen deutlichen Performancesprung.
Beide Agenten haben große Varianz (Std ≈105). Die Ausreißer nach oben zeigen, dass beiden Agenten gelegentlich Glück haben, aber nicht konsistent.
## initialer Ansatz

Für den Initialen Ansatz habe ich PPO (proximal policy optimization) gewählt. Bei PPO kommen zwei Komponenten zum Einsatz: der „Actor“, der die Policy direkt optimiert, und der „Critic“, der den erwarteten Wert über eine Advantage-Funktion schätzt. So verbindet PPO die Stärke von Policy- und Value-Methoden in einem stabilen Lernverfahren.

Ich habe mich für PPO entschieden, weil dies auch verwendet wird, um CHatGPT zu trainieren. Auch im Paper "Deep Q-Learning versus Proximal Policy
Optimization" (https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10228056) wird gezeigt, dass PPO in vielen Umgebungen eine bessere Leistung erbringt als DQN.
### Wrapper
| Wrapper                                | Konfiguration                                                          |
| -------------------------------------- | ---------------------------------------------------------------------- |
| `gym.wrappers.RecordVideo`             | `f"videos/{run_name}"`, `render_mode="rgb_array"` (nur für `idx == 0`) |
| `gym.wrappers.RecordEpisodeStatistics` | —                                                                      |
| `NoopResetEnv`                         | `noop_max=30`                                                          |
| `MaxAndSkipEnv`                        | `skip=args.skip_frames`                                                |
| `EpisodicLifeEnv`                      | —                                                                      |
| `FireResetEnv`                         | — (wenn `"FIRE"` in `env.unwrapped.get_action_meanings()`)             |
| `ClipRewardEnv`                        | —                                                                      |
| `gym.wrappers.ResizeObservation`       | `shape=(84, 84)`                                                       |
| `gym.wrappers.GrayscaleObservation`    | —                                                                      |
| `gym.wrappers.FrameStackObservation`   | `num_stack=4`                                                          |

### Hyperparameter
| Hyperparameter       | Wert                                       | Beschreibung                                                                                                                            |
| -------------------- | ------------------------------------------ |-----------------------------------------------------------------------------------------------------------------------------------------|
| exp\_name            | `os.path.basename(__file__)[:-len(".py")]` | name des experiment                                                                                                                     |
| seed                 | 1                                          | seed                                                                                                                                    |
| env\_id              | `"ALE/SpaceInvaders-v5"`                   | Umgebung indem der Agent trainiert wird                                                                                                 |
| eval\_episodes       | 100                                        | Anzahl der Episoden für die Evaluation des Agenten                                                                                      |
| wandb\_tracking      | true                                       | WandB-Tracking aktivieren                                                                                                               |
| wandb\_proj          | `"rle-initial"`                            | Name des WandB-Projekts                                                                                                                 |
| wandb\_entity        | null                                       | Name der WandB-Entity (optional)                                                                                                        |
| video                | true                                       | Videos speichern                                                                                                                        |
| torch\_deterministic | true                                       | PyTorch deterministisch                                                                                                                 |
| cuda                 | true                                       | CUDA verwendet werden soll                                                                                                              |
| save\_model          | true                                       | Model speichern                                                                                                                         |
| eval\_checkpoint     | null                                       | Pfad zum gespeicherten Modell für die Evaluation (optional)                                                                             |
| total\_timesteps     | 5\_000\_000                                | Gesamte Anzahl der Trainingsschritte (Frames)                                                                                           |
| lr                   | 2.5e-4                                     | Lernrate für den Optimierer (Adam)                                                                                                      |
| num\_envs            | 16                                         | Anzahl der parallelen Umgebungen                                                                                                        |
| num\_steps           | 128                                        | Anzahl der Schritte pro Update (Rollout-Länge)                                                                                          |
| anneal\_lr           | true                                       | Lernrate anpassen (annealing)                                                                                                           |
| gamma                | 0.99                                       | Discount-Faktor für zukünftige Rewards                                                                                                  |
| gae\_lambda          | 0.95                                       | Lambda für Generalized Advantage Estimation (GAE)                                                                                       |
| num\_minibatches     | 4                                          | Anzahl der Mini-Batches pro Update                                                                                                      |
| update\_epochs       | 4                                          | Anzahl der Update-Epochen pro Mini-Batch                                                                                                |
| norm\_adv            | true                                       | Advantage normalisieren                                                                                                                 |
| clip\_coef           | 0.1                                        | Clipping-Koeffizient für die Policy-Gradient-Berechnung                                                                                 |
| clip\_vloss          | true                                       | Clipping der Value-Loss-Berechnung                                                                                                      |
| ent\_coef            | 0.01                                       | Koeffizient für den Entropie-Bonus zur Förderung der Exploration                                                                        |
| vf\_coef             | 0.5                                        | Koeffizient für den Value-Loss                                                                                                          |
| max\_grad\_norm      | 0.5                                        | Maximale Gradienten-Norm für Gradient Clipping                                                                                          |
| target\_kl           | null                                       | Ziel-KL-Divergenz für die Policy-Optimierung (optional)                                                                                 |
| skip\_frames         | 4                                          | Anzahl der Frames, die übersprungen werden, um die Aktion zu wählen                                                                     |
| batch\_size          | 0                                          | Batchgröße für die Optimierung (0 da Batchgröße automatisch durch num_envs und num_steps berechnet)                                     |
| minibatch\_size      | 0                                          | Größe der Mini-Batches für die Optimierung (0 da Mini-Batch-Größe automatisch durch num_envs, num_steps und num_minibatches berechnet)  |
| num\_iterations      | 0                                          | Anzahl der Iterationen für die Optimierung (0 da automatisch durch total_timesteps, num_envs und num_steps berechnet)                   |

### Agent
Der Agent in diesem Script besteht aus einem kleinen, konvolutionalen Neuronalen Netzwerk, das drei aufeinanderfolgende Conv2D-Schichten mit jeweils ReLU-Aktivierung einsetzt (Filtergrößen 8×8 mit Stride 4, 4×4 mit Stride 2 und 3×3 mit Stride 1),
gefolgt von einem Flattening und einer voll verbundenen Schicht mit 512 Einheiten. Auf diesen gemeinsamen Repräsentationskopf bauen zwei separate Köpfe auf: Der Actor ist eine lineare Schicht, die auf der 512-dimensionalen Repräsentation die Aktionslogits erzeugt und daraus eine kategoriale Verteilung konstruiert,
um Aktionen zu sampeln. Der Critic ist ebenfalls eine lineare Schicht, die einen einzigen Wert (State-Value) vorhersagt.
Alle linearen und Faltungs-Layer werden orthogonal initialisiert, um stabiles Lernen zu fördern. Während der Interaktion normalisiert das Netzwerk die Eingabebilder durch Division durch 255, um Werte im Bereich [0,1] zu erhalten.

### Ergebnisse

#### Episodenlänge
| Ansatz    | Mittelwert | Standartabweichung |
|-----------|------------|--------------------|
| Initial   | 942.54     | 249.97             |
| heuristic | 663.00     | 182.94             |
| random    | 535.56     | 169.96             |

<img src="plots/init_ep_length_bp.png" alt="Episode length plot" height="120" /><img src="plots/base_heur_ep_length_bp.png" alt="Episode length plot" height="120" /><img src="plots/base_rand_ep_length_bp.png" alt="Episode length plot" height="120" />

Der trainierte Agent erreicht mit im Schnitt 942 Schritten pro Episode eine deutlich längere Lebensdauer als die Heuristik (663) und die zufällige Politik (536). Die größere Standardabweichung (≈250) und die Ausreißer bis über 1 600 Schritte deuten darauf hin, dass das Modell in vielen, aber nicht allen Fällen sehr robust agiert und gelegentlich außergewöhnlich lange durchs Spiel kommt. Die Heuristik bleibt mit moderater Variabilität (σ≈183) im Mittelfeld, während Random mit σ≈170 am wenigsten konsistent ist.

#### Episodic Return
| Ansatz    | Mittelwert | Standartabweichung |
|-----------|------------|--------------------|
| Initial   | 942.54     | 249.97             |
| heuristic | 265.40     | 105.05             |
| random    | 146.60     | 106.41             |

<img src="plots/init_ep_return_bp.png" alt="Episode return plot" height="120" /><img src="plots/base_heur_ep_return_bp.png" alt="Episode return plot" height="120" /><img src="plots/base_rand_ep_return_bp.png" alt="Episode return plot" height="120" />

Bei der kumulierten Belohnung glänzt der gelernte Agent mit durchschnittlich 942 Punkten, das sind fast viermal so viele wie die Heuristik (265) und mehr als sechsfache Belohnung gegenüber Random (147). Auch hier zeigt der Agent die größte Streuung in den Returns (σ≈250), was darauf hindeutet, dass er zwar oft sehr hohe Punktzahlen erzielt, gelegentlich aber auch unterdurchschnittlich abschneidet. Heuristik und Random liefern vergleichsweise schmalere Verteilungen und niedrigere Maxima.

#### Episodic Time
| Ansatz    | Mittelwert | Standartabweichung |
|-----------|------------|--------------------|
| Initial   | 0.61       | 0.16               |
| heuristic | 0.68       | 0.22               |
| random    | 0.59       | 0.19               |
<img src="plots/init_ep_time_bp.png" alt="Episode time plot" height="120" /><img src="plots/base_heur_ep_time_bp.png" alt="Episode time plot" height="120" /><img src="plots/base_rand_ep_time_bp.png" alt="Episode time plot" height="120" />

Die Ausführungszeit pro Episode liegt beim Agenten bei etwa 0,61 s, was nur minimal langsamer ist als Random (0,59 s) und sogar etwas flotter als die Heuristik (0,68 s). Die geringe Varianz in den Laufzeiten (Agent σ≈0,16 s, Heuristik σ≈0,22 s, Random σ≈0,19 s) zeigt, dass die zusätzliche Netzwerkinferenz nur einen sehr moderaten Overhead verursacht und der Agent insgesamt sehr effizient arbeitet.




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
agent gleich wie initial
#### evaluation der Ergebnisse
#### Analyse

## FInaler Vergleich

## Fazit 

## Ausblick