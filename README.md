# Value Iteration per autoscaling cloud

## Descrizione generale

Questo repository contiene un prototipo Python che applica l’algoritmo di **Value Iteration** a un problema di gestione dinamica delle risorse computazionali in ambiente cloud.

Il caso di studio riguarda un agente di autoscaling incaricato di scegliere, a partire dallo stato operativo del servizio, se aumentare, mantenere o ridurre le risorse disponibili. Il problema viene modellato come **Markov Decision Process** e risolto tramite Value Iteration, con l’obiettivo di calcolare una politica ottima che bilanci qualità del servizio, rischio di degrado e costo delle risorse.

Il prototipo non controlla un’infrastruttura cloud reale, ma implementa effettivamente l’algoritmo di Value Iteration su un MDP semplificato e interamente specificato.

## Obiettivo del progetto

L’obiettivo del progetto è mostrare in modo eseguibile come una metodologia di programmazione dinamica possa essere applicata a un problema di decisione sequenziale in ambiente incerto.

In particolare, il programma consente di:

- rappresentare un problema di autoscaling come Markov Decision Process;
- definire stati, azioni, transizioni probabilistiche e funzione di rinforzo;
- applicare la Value Iteration per stimare la value function ottima;
- ricavare la politica ottima a partire dai valori finali degli stati;
- confrontare il valore atteso delle azioni disponibili in ciascuno stato;
- simulare l’applicazione della politica ottima;
- permettere all’utente di scegliere manualmente le azioni e confrontarle con quelle suggerite dalla politica ottima;
- generare un report testuale dei risultati ottenuti.

## Modello del problema

Il dominio scelto è quello dell’autoscaling cloud. In tale contesto, un servizio applicativo può trovarsi in condizioni operative differenti: può essere stabile, sovradimensionato, sottoposto a carico elevato, degradato o critico.

La decisione dell’agente non può essere valutata soltanto in termini immediati. Aumentare risorse può comportare un costo, ma ridurre il rischio di degrado futuro; ridurre risorse può diminuire lo spreco, ma aumentare il rischio di saturazione; mantenere la configurazione può essere adeguato in condizioni stabili, ma insufficiente in condizioni di carico elevato.

Per questo motivo, il problema è formulato come processo decisionale sequenziale sotto incertezza.

## Formalizzazione come Markov Decision Process

Il modello è definito mediante:

- un insieme finito di stati;
- un insieme finito di azioni;
- una funzione di transizione probabilistica;
- una funzione di rinforzo;
- un fattore di sconto.

### Stati

| Stato | Significato |
|---|---|
| `S1` | Servizio stabile. Il servizio funziona correttamente e le risorse sono adeguate. |
| `S2` | Servizio sovradimensionato. Il servizio funziona correttamente, ma utilizza più risorse del necessario. |
| `S3` | Carico elevato. Il traffico è alto, ma il servizio è ancora stabile. |
| `S4` | Servizio degradato. Il servizio è lento o vicino alla saturazione. |
| `S5` | Servizio critico. Il servizio è in forte sofferenza o quasi indisponibile. |

### Azioni

| Azione | Significato |
|---|---|
| `scale_up` | Aumentare le risorse computazionali. |
| `keep` | Mantenere la configurazione corrente. |
| `scale_down` | Ridurre le risorse computazionali. |

### Transizioni

Le transizioni sono probabilistiche. Per ogni coppia stato-azione viene indicata una distribuzione di probabilità sugli stati successivi.

Questa scelta rappresenta l’incertezza dell’ambiente: la stessa azione, applicata nello stesso stato, non produce necessariamente sempre lo stesso esito. Ad esempio, aumentare le risorse quando il servizio è degradato può migliorare lo stato del sistema, ma non garantisce necessariamente il ritorno immediato a una condizione completamente stabile.

### Rinforzi

La funzione di rinforzo è definita nella forma:

```text
R(s, a, s')
```

Il rinforzo dipende dallo stato di partenza `s`, dall’azione scelta `a` e dallo stato raggiunto `s'`.

La funzione considera due componenti:

- la qualità operativa dello stato raggiunto;
- il costo o il rischio associato all’azione scelta.

In questo modo, il modello evita una politica banale. L’agente non è incentivato ad aumentare sempre le risorse perché tale azione può essere costosa; allo stesso tempo, non è incentivato a ridurle indiscriminatamente perché ciò può aumentare il rischio di degrado o criticità.

## Algoritmo implementato

Il file `value_iteration_cloud_autoscaling.py` implementa l’algoritmo di **Value Iteration**.

La value function viene inizializzata a zero per tutti gli stati. A ogni iterazione, per ciascuno stato, il programma calcola il valore atteso delle azioni disponibili e aggiorna il valore dello stato scegliendo il massimo tra tali valori. In questo modo, l’algoritmo approssima progressivamente la value function ottima.

A convergenza, la value function soddisfa l’equazione di Bellman:

$$
V^{\ast}(s)=\max_{a}\sum_{s^{\prime}}P^{a}_{s s^{\prime}}\left[R^{a}_{s s^{\prime}}+\gamma V^{\ast}(s^{\prime})\right]
$$

Una volta stimata la value function ottima, la politica ottima viene ricavata scegliendo, per ogni stato, l’azione che massimizza il valore atteso:

$$
\pi^{\ast}(s)=\arg\max_{a}\sum_{s^{\prime}}P^{a}_{s s^{\prime}}\left[R^{a}_{s s^{\prime}}+\gamma V^{\ast}(s^{\prime})\right]
$$

Il processo iterativo continua fino a quando la massima variazione dei valori degli stati, indicata come $\Delta$, diventa inferiore alla soglia di convergenza $\theta$.

I parametri utilizzati sono:

| Parametro | Valore | Significato |
|---|---:|---|
| $\gamma$ | `0.9` | Fattore di sconto dei rinforzi futuri. |
| $\theta$ | `0.001` | Soglia di convergenza della value function. |

Il valore $\gamma = 0.9$ attribuisce rilevanza significativa alle conseguenze future delle azioni. Il valore $\theta = 0.001$ stabilisce il criterio di arresto dell’algoritmo.

## Risultati ottenuti

L’esecuzione del programma porta alla convergenza della Value Iteration in 87 iterazioni con $\Delta$ finale pari a circa `0.000918`.

La politica ottima calcolata è la seguente:

| Stato | Azione ottima |
|---|---|
| `S1` | `keep` |
| `S2` | `scale_down` |
| `S3` | `scale_up` |
| `S4` | `scale_up` |
| `S5` | `scale_up` |

Il risultato è coerente con il significato del dominio:

- nello stato stabile, l’agente mantiene la configurazione;
- nello stato sovradimensionato, l’agente riduce le risorse;
- negli stati di carico elevato, degrado o criticità, l’agente aumenta le risorse.

La politica ottenuta non dipende da una scelta manuale fissata nel codice, ma deriva dal confronto tra i valori attesi delle azioni, calcolati tramite la value function finale.

## Struttura del repository

| File | Descrizione |
|---|---|
| `value_iteration_cloud_autoscaling.py` | Programma principale. Contiene il modello MDP, l’algoritmo di Value Iteration, la simulazione e la modalità manuale. |
| `report_value_iteration.txt` | Report generato dal programma con parametri, value function, politica ottima, valori attesi delle azioni e interpretazione sintetica. |
| `README.md` | Documentazione del progetto. |

## Requisiti

Il programma utilizza esclusivamente moduli standard di Python. Non è, quindi, necessario installare librerie esterne.

Versione utilizzata durante lo sviluppo e il test:

```text
Python 3.12.10
```

## Esecuzione

Aprire un terminale nella cartella del progetto.

Su Windows PowerShell, ad esempio:

```powershell
cd C:\Users\Hp\Desktop\value-iteration-cloud-autoscaling
```

Eseguire poi:

```powershell
python value_iteration_cloud_autoscaling.py
```

All’avvio viene mostrato un menu interattivo.

```text
1. Mostra stati e azioni
2. Mostra modello di transizione ed esempi di rinforzo
3. Esegui Value Iteration
4. Mostra value function finale
5. Mostra politica ottima
6. Mostra valori attesi delle azioni
7. Mostra convergenza
8. Simula la politica ottima da uno stato iniziale
9. Modalità manuale: scegli azioni e confrontale con la politica ottima
10. Genera report report_value_iteration.txt
0. Esci
```

## Sequenza di utilizzo consigliata

Per osservare i risultati principali è possibile eseguire le seguenti opzioni:

```text
3
4
5
6
7
10
0
```

La sequenza consente di:

- eseguire la Value Iteration;
- visualizzare la value function finale;
- visualizzare la politica ottima;
- confrontare i valori attesi delle azioni;
- osservare la convergenza dell’algoritmo;
- generare il report testuale;
- terminare il programma.

## Modalità manuale

La modalità manuale, accessibile tramite l’opzione `9`, permette all’utente di scegliere direttamente le azioni dell’agente.

A ogni passo il programma mostra:

- lo stato corrente;
- l’azione suggerita dalla politica ottima;
- l’azione scelta manualmente;
- lo stato successivo raggiunto;
- il rinforzo immediato ottenuto;
- il confronto tra scelta manuale e azione ottima.

Questa modalità non modifica la politica calcolata dall’algoritmo. Serve, invece, a mostrare, in modo operativo, come una scelta manuale possa coincidere o meno con la politica ottima ottenuta tramite Value Iteration.

## Report dei risultati

Selezionando l’opzione `10`, il programma genera il file:

```text
report_value_iteration.txt
```

Il report contiene:

- obiettivo del prototipo;
- stati del modello;
- azioni disponibili;
- parametri della Value Iteration;
- numero di iterazioni;
- $\Delta$ finale;
- value function finale;
- politica ottima;
- valori attesi delle azioni;
- interpretazione sintetica dei risultati.

Il report ha funzione documentale: consente di consultare i risultati principali anche senza rieseguire il programma.

## Nota metodologica

Il prototipo implementa realmente l’algoritmo di Value Iteration, ma lo applica a un modello MDP semplificato del problema di autoscaling cloud.

Le transizioni e i rinforzi non derivano da misurazioni reali su un provider cloud, ma sono definiti nel codice per costruire un caso di studio coerente, interpretabile e adatto a mostrare il funzionamento dell’algoritmo.

Le simulazioni prodotte dalle opzioni `8` e `9` rappresentano traiettorie probabilistiche generate dal modello, non l’evoluzione di un’infrastruttura cloud reale.

## Nota conclusiva

Il prototipo non vuole essere un sistema industriale completo per l’autoscaling cloud, ma una realizzazione didattica coerente con il caso di studio discusso nell’elaborato scritto.

Il suo scopo è mostrare in forma eseguibile l’applicazione della Value Iteration a un Markov Decision Process definito in un dominio ristretto ma significativo. Il programma consente, infatti, di osservare come, a partire da stati, azioni, transizioni probabilistiche e rinforzi, sia possibile calcolare una value function, ricavare una politica ottima e confrontare tale politica con scelte manuali effettuate dall’utente.

Il progetto rende, pertanto, verificabile l’intero passaggio:

```text
problema di autoscaling cloud
→ astrazione in stati e azioni
→ definizione di transizioni probabilistiche e rinforzi
→ applicazione della Value Iteration
→ calcolo della value function
→ estrazione della politica ottima
→ simulazione e interpretazione dei risultati
```

In questo senso, il valore del prototipo non risiede nella complessità tecnica dell’infrastruttura simulata, ma nella chiarezza con cui rende osservabile il passaggio dal modello teorico del Reinforcement Learning alla sua applicazione computazionale.
