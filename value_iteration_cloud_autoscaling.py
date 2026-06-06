"""
Prototipo: Value Iteration per un agente di autoscaling cloud.

Il programma modella un servizio cloud semplificato come Markov Decision Process.
L'agente osserva lo stato operativo del servizio e sceglie se aumentare, mantenere
o ridurre le risorse computazionali.

L'obiettivo è mostrare in modo eseguibile come la Value Iteration possa calcolare
una politica ottima, cioè una regola che associa a ogni stato l'azione più conveniente
nel lungo periodo.

Il programma permette di:
visualizzare il modello MDP;
eseguire la Value Iteration;
osservare la value function finale;
osservare la politica ottima;
confrontare i valori attesi delle azioni;
simulare la politica ottima;
scegliere azioni manualmente e confrontarle con la politica ottima;
salvare un report testuale leggibile nel file report_value_iteration.txt.
"""

import random
from typing import Dict, List, Tuple


State = str
Action = str
ValueFunction = Dict[State, float]
Policy = Dict[State, Action]


GAMMA = 0.9
THETA = 0.001
MAX_ITERATIONS = 10_000
REPORT_FILENAME = "report_value_iteration.txt"

RANDOM_SEED = 42


STATES: Dict[State, str] = {
    "S1": "Servizio stabile. Il servizio funziona bene e le risorse sono adeguate.",
    "S2": "Servizio sovradimensionato. Il servizio funziona bene, ma usa troppe risorse.",
    "S3": "Carico elevato. Il traffico è alto, ma il servizio è ancora stabile.",
    "S4": "Servizio degradato. Il servizio è lento o vicino alla saturazione.",
    "S5": "Servizio critico. Il servizio è in forte sofferenza o quasi indisponibile.",
}


ACTIONS: Dict[Action, str] = {
    "scale_up": "Aumentare le risorse computazionali.",
    "keep": "Mantenere la configurazione attuale.",
    "scale_down": "Ridurre le risorse computazionali.",
}


STATE_ORDER: List[State] = ["S1", "S2", "S3", "S4", "S5"]
ACTION_ORDER: List[Action] = ["scale_up", "keep", "scale_down"]


"""
Le transizioni descrivono l'incertezza dell'ambiente.

Per ogni stato e per ogni azione, viene indicata una distribuzione di probabilità
sugli stati successivi. Ad esempio, se il servizio è degradato e l'agente sceglie
scale_up, il sistema può migliorare, ma non è garantito che torni subito stabile.

La somma delle probabilità associate a ogni coppia stato-azione deve essere uguale a 1.
"""

TRANSITIONS: Dict[State, Dict[Action, Dict[State, float]]] = {
    "S1": {
        "keep": {"S1": 0.75, "S3": 0.20, "S2": 0.05},
        "scale_up": {"S2": 0.70, "S1": 0.25, "S3": 0.05},
        "scale_down": {"S1": 0.55, "S3": 0.25, "S2": 0.15, "S4": 0.05},
    },
    "S2": {
        "keep": {"S2": 0.70, "S1": 0.25, "S3": 0.05},
        "scale_up": {"S2": 0.90, "S1": 0.10},
        "scale_down": {"S1": 0.75, "S3": 0.15, "S2": 0.10},
    },
    "S3": {
        "keep": {"S3": 0.50, "S4": 0.35, "S1": 0.15},
        "scale_up": {"S1": 0.60, "S3": 0.30, "S2": 0.10},
        "scale_down": {"S4": 0.55, "S5": 0.25, "S3": 0.20},
    },
    "S4": {
        "keep": {"S4": 0.50, "S5": 0.30, "S3": 0.20},
        "scale_up": {"S3": 0.55, "S1": 0.25, "S4": 0.20},
        "scale_down": {"S5": 0.75, "S4": 0.25},
    },
    "S5": {
        "keep": {"S5": 0.60, "S4": 0.30, "S3": 0.10},
        "scale_up": {"S4": 0.55, "S3": 0.30, "S1": 0.15},
        "scale_down": {"S5": 0.90, "S4": 0.10},
    },
}


"""
STATE_QUALITY assegna un valore allo stato raggiunto.

Gli stati desiderabili ricevono valori positivi, mentre gli stati degradati o critici
ricevono valori negativi. Questo rappresenta la qualità operativa del servizio cloud.
"""

STATE_QUALITY: Dict[State, int] = {
    "S1": 10,
    "S2": 4,
    "S3": 3,
    "S4": -8,
    "S5": -20,
}


def action_adjustment(state: State, action: Action) -> int:
    """
    Calcola la correzione del rinforzo legata all'azione scelta.

    L'azione scale_up ha un costo, perché aumentare risorse cloud richiede maggiore spesa.
    Questo costo è più grave se il servizio è già stabile o sovradimensionato.

    L'azione scale_down può essere positiva se il servizio è sovradimensionato,
    perché riduce lo spreco di risorse. Diventa invece rischiosa se il servizio
    è già sotto carico, degradato o critico.
    """
    if action == "keep":
        return 0

    if action == "scale_up":
        if state in {"S1", "S2"}:
            return -5
        return -2

    if action == "scale_down":
        if state == "S2":
            return 4
        if state == "S1":
            return -1
        return -6

    raise ValueError(f"Azione non riconosciuta: {action}")


def reward(state: State, action: Action, next_state: State) -> int:
    """
    Calcola il rinforzo R(s, a, s').

    Il rinforzo dipende da due elementi:
    la qualità dello stato raggiunto;
    il costo o il rischio dell'azione scelta nello stato di partenza.

    Questa formulazione rende il caso più realistico: non basta arrivare in uno
    stato buono, ma bisogna anche considerare se l'azione usata per arrivarci
    è costosa, prudente o rischiosa.
    """
    return STATE_QUALITY[next_state] + action_adjustment(state, action)


def validate_mdp() -> None:
    """
    Controlla che il modello MDP sia definito correttamente.

    Il controllo è utile perché la Value Iteration presuppone un MDP valido.
    In particolare, per ogni coppia stato-azione, le probabilità degli stati
    successivi devono sommare a 1.
    """
    for state in STATE_ORDER:
        if state not in TRANSITIONS:
            raise ValueError(f"Mancano le transizioni per lo stato {state}.")

        for action in ACTION_ORDER:
            if action not in TRANSITIONS[state]:
                raise ValueError(f"Manca l'azione {action} nello stato {state}.")

            probability_sum = sum(TRANSITIONS[state][action].values())

            if abs(probability_sum - 1.0) > 1e-9:
                raise ValueError(
                    f"Le probabilità per ({state}, {action}) sommano a {probability_sum}, non a 1."
                )

            for next_state in TRANSITIONS[state][action]:
                if next_state not in STATES:
                    raise ValueError(f"Stato successivo non riconosciuto: {next_state}.")


def expected_return(state: State, action: Action, values: ValueFunction, gamma: float) -> float:
    """
    Calcola il valore atteso di una certa azione in uno stato.

    Questa funzione implementa il nucleo dell'equazione di Bellman.
    Per ogni stato successivo possibile considera:
    probabilità della transizione;
    rinforzo immediato;
    valore futuro dello stato successivo, pesato dal fattore di sconto gamma.
    """
    total = 0.0

    for next_state, probability in TRANSITIONS[state][action].items():
        immediate_reward = reward(state, action, next_state)
        discounted_future_value = gamma * values[next_state]
        total += probability * (immediate_reward + discounted_future_value)

    return total


def value_iteration(
    gamma: float = GAMMA,
    theta: float = THETA,
    max_iterations: int = MAX_ITERATIONS,
) -> Tuple[ValueFunction, Policy, int, List[float]]:
    """
    Esegue l'algoritmo di Value Iteration.

    La value function viene inizializzata a zero per tutti gli stati.
    A ogni iterazione, il valore di ciascuno stato viene aggiornato scegliendo
    l'azione che massimizza il rinforzo scontato atteso.

    L'algoritmo termina quando la massima variazione dei valori, indicata con delta,
    diventa minore della soglia theta.
    """
    values: ValueFunction = {state: 0.0 for state in STATE_ORDER}
    delta_history: List[float] = []

    for iteration in range(1, max_iterations + 1):
        old_values = values.copy()
        delta = 0.0

        for state in STATE_ORDER:
            action_values = [
                expected_return(state, action, old_values, gamma)
                for action in ACTION_ORDER
            ]

            best_value = max(action_values)
            values[state] = best_value
            delta = max(delta, abs(best_value - old_values[state]))

        delta_history.append(delta)

        if delta < theta:
            break

    policy = extract_policy(values, gamma)
    return values, policy, iteration, delta_history


def extract_policy(values: ValueFunction, gamma: float = GAMMA) -> Policy:
    """
    Ricava la politica ottima dalla value function finale.

    Per ogni stato si calcola il valore atteso di tutte le azioni disponibili.
    L'azione con valore atteso massimo diventa l'azione indicata dalla politica.
    """
    policy: Policy = {}

    for state in STATE_ORDER:
        best_action = max(
            ACTION_ORDER,
            key=lambda action: expected_return(state, action, values, gamma),
        )
        policy[state] = best_action

    return policy


def print_blank_lines() -> None:
    """Aggiunge spazio nel terminale senza usare linee grafiche di separazione."""
    print()
    print()


def print_title(title: str) -> None:
    """Mostra un titolo semplice, senza linee decorative."""
    print_blank_lines()
    print(title.upper())
    print()


def print_states_and_actions() -> None:
    """
    Mostra stati e azioni del modello.

    Questa opzione serve a comprendere come il problema reale dell'autoscaling
    sia stato trasformato in una rappresentazione finita, adatta alla Value Iteration.
    """
    print_title("Stati e azioni del modello")

    print("Questa sezione mostra l'astrazione del servizio cloud.")
    print("Gli stati rappresentano le condizioni operative del servizio.")
    print("Le azioni rappresentano le scelte disponibili per l'agente di autoscaling.")

    print()
    print("Stati")

    for state in STATE_ORDER:
        print(f"{state}: {STATES[state]}")

    print()
    print("Azioni")

    for action in ACTION_ORDER:
        print(f"{action}: {ACTIONS[action]}")


def print_transition_model() -> None:
    """
    Mostra le transizioni probabilistiche dell'MDP.

    Le transizioni indicano che cosa può accadere dopo una certa azione.
    Poiché l'ambiente è incerto, la stessa azione può condurre a più stati successivi.
    """
    print_title("Modello di transizione")

    print("Per ogni stato e per ogni azione vengono mostrate le probabilità degli stati successivi.")
    print("Questa è la parte del modello che rappresenta l'incertezza dell'ambiente.")

    for state in STATE_ORDER:
        print()
        print(f"{state}: {STATES[state]}")

        for action in ACTION_ORDER:
            parts = [
                f"{next_state} con probabilità {probability:.2f}"
                for next_state, probability in TRANSITIONS[state][action].items()
            ]
            print(f"{action}: " + "; ".join(parts))


def print_reward_examples() -> None:
    """
    Mostra alcuni esempi della funzione di rinforzo.

    Gli esempi aiutano a capire che il rinforzo non dipende solo dallo stato raggiunto,
    ma anche dal costo o dal rischio dell'azione scelta.
    """
    examples = [
        ("S3", "scale_up", "S1"),
        ("S4", "scale_up", "S3"),
        ("S2", "scale_down", "S1"),
        ("S4", "scale_down", "S5"),
        ("S1", "scale_up", "S2"),
    ]

    print()
    print("Esempi di rinforzo")

    print("La forma usata è R(stato di partenza, azione, stato raggiunto).")
    print("Un valore positivo indica un esito favorevole; un valore negativo indica un esito sfavorevole.")

    for state, action, next_state in examples:
        value = reward(state, action, next_state)
        print(f"R({state}, {action}, {next_state}) = {value}")


def print_value_function(values: ValueFunction) -> None:
    """
    Mostra la value function finale.

    I valori non rappresentano il rinforzo immediato di una singola azione.
    Rappresentano il rinforzo scontato atteso nel lungo periodo partendo da ciascuno stato.
    """
    print_title("Value function finale")

    print("La value function assegna un valore numerico a ogni stato.")
    print("Valori più alti indicano stati da cui l'agente può ottenere un ritorno atteso maggiore.")
    print("I valori sono elevati perché incorporano anche le ricompense future scontate.")

    print()

    for state in STATE_ORDER:
        print(f"{state}: {values[state]:.4f}")


def print_policy(policy: Policy) -> None:
    """
    Mostra la politica ottima calcolata dall'algoritmo.

    La politica ottima indica quale azione scegliere in ogni stato.
    È il risultato principale della Value Iteration.
    """
    print_title("Politica ottima")

    print("La politica ottima associa a ogni stato l'azione con massimo valore atteso.")
    print("Questa tabella è la regola di comportamento finale dell'agente di autoscaling.")

    print()

    for state in STATE_ORDER:
        action = policy[state]
        print(f"{state}: {action}. {ACTIONS[action]}")


def print_action_values(values: ValueFunction, gamma: float = GAMMA) -> None:
    """
    Mostra il valore atteso di ogni azione in ogni stato.

    Questa opzione consente di verificare perché una certa azione viene scelta
    dalla politica ottima. L'azione selezionata è quella con valore atteso massimo.
    """
    print_title("Valori attesi delle azioni")

    print("Per ogni stato vengono confrontate le tre azioni disponibili.")
    print("Il valore più alto corrisponde all'azione scelta dalla politica ottima.")

    for state in STATE_ORDER:
        print()
        print(f"{state}: {STATES[state]}")

        action_values = {
            action: expected_return(state, action, values, gamma)
            for action in ACTION_ORDER
        }

        best_action = max(action_values, key=action_values.get)

        for action in ACTION_ORDER:
            marker = "azione ottima" if action == best_action else "alternativa"
            print(f"{action}: {action_values[action]:.4f}. {marker}")


def print_convergence(delta_history: List[float]) -> None:
    """
    Mostra una sintesi della convergenza.

    Delta misura quanto cambia la value function tra un'iterazione e la successiva.
    Quando delta diventa minore di theta, l'algoritmo si arresta.
    """
    print_title("Convergenza della Value Iteration")

    print("Delta misura la massima variazione dei valori degli stati tra due iterazioni consecutive.")
    print(f"La soglia di arresto theta è {THETA}.")
    print("Quando delta scende sotto questa soglia, la value function è considerata stabile.")

    print()
    print(f"Iterazioni eseguite: {len(delta_history)}")
    print(f"Delta finale: {delta_history[-1]:.6f}")

    print()
    print("Prime iterazioni")

    for index, delta in enumerate(delta_history[:10], start=1):
        print(f"Iterazione {index}: delta = {delta:.6f}")

    if len(delta_history) > 10:
        print()
        print("Ultima iterazione")
        print(f"Iterazione {len(delta_history)}: delta = {delta_history[-1]:.6f}")


def choose_next_state(state: State, action: Action) -> State:
    """
    Estrae casualmente uno stato successivo secondo le probabilità di transizione.

    Questa funzione viene usata nelle simulazioni. Serve a mostrare che l'ambiente
    non è deterministico: la stessa azione può produrre esiti diversi.
    """
    roll = random.random()
    cumulative_probability = 0.0

    for next_state, probability in TRANSITIONS[state][action].items():
        cumulative_probability += probability

        if roll <= cumulative_probability:
            return next_state

    return list(TRANSITIONS[state][action].keys())[-1]


def ask_state() -> State:
    """
    Chiede all'utente di selezionare uno stato iniziale.

    L'utente può inserire sia il numero dello stato sia la sigla dello stato.
    Sono quindi accettati input come 1 oppure S1.
    """
    while True:
        print()
        print("Scegli uno stato iniziale.")
        print("Puoi inserire il numero dello stato oppure la sigla, ad esempio 1 oppure S1.")

        for index, state in enumerate(STATE_ORDER, start=1):
            print(f"{index}. {state}: {STATES[state]}")

        choice = input("Stato iniziale: ").strip().upper()

        if choice.isdigit():
            index = int(choice)

            if 1 <= index <= len(STATE_ORDER):
                return STATE_ORDER[index - 1]

        if choice in STATE_ORDER:
            return choice

        print("Scelta non valida. Inserisci un numero tra 1 e 5 oppure una sigla tra S1 e S5.")


def ask_action() -> Action:
    """
    Chiede all'utente di selezionare un'azione.

    L'utente può inserire il numero dell'azione oppure il nome dell'azione.
    Sono quindi accettati input come 1, scale_up, keep oppure scale_down.
    """
    aliases = {
        "1": "scale_up",
        "2": "keep",
        "3": "scale_down",
        "UP": "scale_up",
        "SCALE_UP": "scale_up",
        "KEEP": "keep",
        "DOWN": "scale_down",
        "SCALE_DOWN": "scale_down",
    }

    while True:
        print()
        print("Scegli un'azione.")
        print("Puoi inserire il numero dell'azione oppure il nome, ad esempio 1 oppure scale_up.")

        for index, action in enumerate(ACTION_ORDER, start=1):
            print(f"{index}. {action}: {ACTIONS[action]}")

        choice = input("Azione scelta: ").strip().upper()

        if choice in aliases:
            return aliases[choice]

        print("Scelta non valida. Inserisci 1, 2, 3 oppure scale_up, keep, scale_down.")


def ask_steps(default_steps: int = 8) -> int:
    """
    Chiede il numero di passi della simulazione.

    Un passo corrisponde a una decisione dell'agente:
    stato attuale, azione scelta, nuovo stato e rinforzo ottenuto.
    """
    choice = input(
        f"Numero di passi da simulare. Premi Invio per usare {default_steps} passi: "
    ).strip()

    if not choice:
        return default_steps

    if choice.isdigit() and int(choice) > 0:
        return int(choice)

    print(f"Valore non valido. Uso il valore predefinito: {default_steps} passi.")
    return default_steps


def simulate_optimal_policy(policy: Policy) -> None:
    """
    Simula l'esecuzione della politica ottima.

    L'utente sceglie lo stato iniziale e il numero di passi.
    A ogni passo il programma applica automaticamente l'azione indicata dalla politica ottima.
    """
    print_title("Simulazione della politica ottima")

    print("Questa simulazione mostra come si comporta l'agente seguendo sempre la politica ottima.")
    print("Le transizioni sono probabilistiche, quindi esecuzioni diverse possono produrre traiettorie diverse.")

    state = ask_state()
    steps = ask_steps()

    total_reward = 0

    for step in range(1, steps + 1):
        action = policy[state]
        next_state = choose_next_state(state, action)
        immediate_reward = reward(state, action, next_state)
        total_reward += immediate_reward

        print()
        print(f"Passo {step} di {steps}")
        print(f"Stato attuale: {state}: {STATES[state]}")
        print(f"Azione scelta dalla politica ottima: {action}")
        print(f"Nuovo stato raggiunto: {next_state}: {STATES[next_state]}")
        print(f"Rinforzo immediato ottenuto: {immediate_reward}")

        state = next_state

    print()
    print(f"Rinforzo totale ottenuto nella simulazione: {total_reward}")
    print("Questo valore è la somma dei rinforzi immediati osservati nella simulazione.")


def manual_mode(policy: Policy) -> None:
    """
    Permette all'utente di scegliere manualmente le azioni.

    Dopo ogni scelta, il programma mostra:
    lo stato di partenza;
    l'azione scelta dall'utente;
    l'azione suggerita dalla politica ottima;
    lo stato raggiunto;
    il rinforzo immediato ottenuto.

    Questa modalità serve a verificare concretamente il significato della politica ottima.
    """
    print_title("Modalità manuale")

    print("In questa modalità l'utente sceglie direttamente le azioni dell'agente.")
    print("A ogni passo il programma confronta la scelta manuale con la politica ottima calcolata dalla Value Iteration.")
    print("Il nuovo stato viene estratto secondo le probabilità di transizione dell'MDP.")
    print("Per questo motivo la simulazione rappresenta un ambiente incerto, non una sequenza deterministica.")

    state = ask_state()
    steps = ask_steps(default_steps=5)

    total_reward = 0

    for step in range(1, steps + 1):
        optimal_action = policy[state]

        print()
        print(f"Passo {step} di {steps}")
        print(f"Stato attuale: {state}: {STATES[state]}")
        print(f"Azione suggerita dalla politica ottima: {optimal_action}")

        chosen_action = ask_action()
        next_state = choose_next_state(state, chosen_action)
        immediate_reward = reward(state, chosen_action, next_state)
        total_reward += immediate_reward

        print()
        print("Risultato del passo")
        print(f"Azione scelta manualmente: {chosen_action}")
        print(f"Nuovo stato raggiunto: {next_state}: {STATES[next_state]}")
        print(f"Rinforzo immediato ottenuto: {immediate_reward}")

        if chosen_action == optimal_action:
            print("Confronto con la politica ottima: la scelta manuale coincide con l'azione ottima.")
        else:
            print(f"Confronto con la politica ottima: l'azione ottima sarebbe stata {optimal_action}.")

        state = next_state

    print()
    print(f"Rinforzo totale ottenuto nella simulazione manuale: {total_reward}")
    print("Questo valore è la somma dei rinforzi immediati osservati nella simulazione, non la value function dello stato iniziale.")


def build_report(
    values: ValueFunction,
    policy: Policy,
    iterations: int,
    delta_history: List[float],
) -> List[str]:
    """
    Costruisce il contenuto del report report_value_iteration.txt.

    Il report è pensato per essere leggibile anche senza eseguire il programma.
    Riassume il caso di studio, il modello MDP, i parametri dell'algoritmo,
    la value function finale, la politica ottima e l'interpretazione dei risultati.
    """
    lines: List[str] = []

    lines.append("VALUE ITERATION PER UN AGENTE DI AUTOSCALING CLOUD\n\n")

    lines.append("1. Obiettivo del prototipo\n\n")
    lines.append(
        "Il prototipo modella un servizio cloud come Markov Decision Process e applica "
        "la Value Iteration per calcolare una politica ottima di autoscaling.\n"
    )
    lines.append(
        "L'agente deve decidere se aumentare, mantenere o ridurre le risorse computazionali "
        "in base allo stato operativo del servizio.\n\n"
    )

    lines.append("2. Stati del modello\n\n")
    for state in STATE_ORDER:
        lines.append(f"{state}: {STATES[state]}\n")

    lines.append("\n3. Azioni disponibili\n\n")
    for action in ACTION_ORDER:
        lines.append(f"{action}: {ACTIONS[action]}\n")

    lines.append("\n4. Parametri della Value Iteration\n\n")
    lines.append(f"Gamma: {GAMMA}\n")
    lines.append(
        "Gamma è il fattore di sconto. Un valore pari a 0.9 attribuisce forte importanza "
        "alle conseguenze future delle azioni.\n"
    )
    lines.append(f"Theta: {THETA}\n")
    lines.append(
        "Theta è la soglia di convergenza. L'algoritmo si arresta quando la variazione "
        "massima della value function scende sotto questo valore.\n"
    )
    lines.append(f"Iterazioni eseguite: {iterations}\n")
    lines.append(f"Delta finale: {delta_history[-1]:.6f}\n\n")

    lines.append("5. Value function finale\n\n")
    lines.append(
        "La value function assegna a ogni stato il rinforzo scontato atteso partendo da quello stato "
        "e seguendo poi la politica ottima. Non rappresenta il rinforzo immediato, ma una stima "
        "del valore complessivo di lungo periodo.\n\n"
    )

    for state in STATE_ORDER:
        lines.append(f"{state}: {values[state]:.4f}\n")

    lines.append("\n6. Politica ottima\n\n")
    lines.append(
        "La politica ottima indica l'azione con valore atteso massimo in ciascuno stato.\n\n"
    )

    for state in STATE_ORDER:
        action = policy[state]
        lines.append(f"{state}: {action}. {ACTIONS[action]}\n")

    lines.append("\n7. Valori attesi delle azioni\n\n")
    lines.append(
        "Per ogni stato vengono riportati i valori attesi delle tre azioni disponibili. "
        "L'azione scelta dalla politica ottima è quella con valore maggiore.\n"
    )

    for state in STATE_ORDER:
        lines.append(f"\n{state}: {STATES[state]}\n")

        action_values = {
            action: expected_return(state, action, values, GAMMA)
            for action in ACTION_ORDER
        }

        best_action = max(action_values, key=action_values.get)

        for action in ACTION_ORDER:
            marker = "azione scelta dalla politica ottima" if action == best_action else "alternativa"
            lines.append(f"{action}: {action_values[action]:.4f}. {marker}.\n")

    lines.append("\n8. Interpretazione sintetica\n\n")
    lines.append(
        "La politica ottenuta è coerente con il significato del dominio. "
        "Quando il servizio è stabile, l'agente mantiene la configurazione. "
        "Quando il servizio è sovradimensionato, riduce le risorse. "
        "Quando il carico è elevato, degradato o critico, aumenta le risorse per ridurre il rischio "
        "di peggioramento futuro.\n\n"
    )
    lines.append(
        "Il risultato mostra come la Value Iteration non scelga semplicemente l'azione con il vantaggio "
        "immediato più evidente, ma tenga conto del valore atteso degli stati futuri.\n"
    )

    return lines


def save_report(
    values: ValueFunction,
    policy: Policy,
    iterations: int,
    delta_history: List[float],
    filename: str = REPORT_FILENAME,
) -> None:
    """
    Salva un report testuale con i risultati della Value Iteration.

    Il file viene creato nella stessa cartella in cui si trova questo programma.
    Il report può essere consultato anche senza rieseguire il codice.
    """
    report_lines = build_report(values, policy, iterations, delta_history)

    with open(filename, "w", encoding="utf-8") as file:
        file.writelines(report_lines)

    print_title("Report generato")

    print(f"Ho creato il file {filename}.")
    print("Lo trovi nella stessa cartella in cui è salvato questo programma Python.")
    print("Il report contiene una spiegazione leggibile del modello MDP e dei risultati calcolati.")
    print("In particolare include stati, azioni, parametri, value function finale, politica ottima e valori attesi delle azioni.")


def compute_if_needed(
    values: ValueFunction | None,
    policy: Policy | None,
    iterations: int | None,
    delta_history: List[float] | None,
) -> Tuple[ValueFunction, Policy, int, List[float]]:
    """
    Esegue la Value Iteration solo se necessario.

    In questo modo l'utente può scegliere direttamente un'opzione del menu
    che richiede i risultati, anche senza avere prima selezionato l'opzione
    di esecuzione dell'algoritmo.
    """
    if values is not None and policy is not None and iterations is not None and delta_history is not None:
        return values, policy, iterations, delta_history

    print()
    print("La Value Iteration non è ancora stata eseguita.")
    print("Eseguo ora l'algoritmo per calcolare value function e politica ottima.")

    return value_iteration()


def print_menu() -> None:
    """
    Mostra il menu principale.

    Il menu consente di esplorare il modello, eseguire l'algoritmo e verificare i risultati.
    """
    print_blank_lines()
    print("VALUE ITERATION PER AUTOSCALING CLOUD")
    print()
    print("Scegli una delle seguenti opzioni.")
    print()
    print("1. Mostra stati e azioni")
    print("2. Mostra modello di transizione ed esempi di rinforzo")
    print("3. Esegui Value Iteration")
    print("4. Mostra value function finale")
    print("5. Mostra politica ottima")
    print("6. Mostra valori attesi delle azioni")
    print("7. Mostra convergenza")
    print("8. Simula la politica ottima da uno stato iniziale")
    print("9. Modalità manuale: scegli azioni e confrontale con la politica ottima")
    print("10. Genera report report_value_iteration.txt")
    print("0. Esci")


def main() -> None:
    """
    Avvia il programma interattivo.

    La funzione mantiene in memoria i risultati della Value Iteration dopo la prima esecuzione.
    Se l'utente richiede risultati prima di eseguire l'algoritmo, il programma lo esegue automaticamente.
    """
    random.seed(RANDOM_SEED)
    validate_mdp()

    values: ValueFunction | None = None
    policy: Policy | None = None
    iterations: int | None = None
    delta_history: List[float] | None = None

    while True:
        print_menu()
        choice = input("Seleziona un'opzione: ").strip()

        if choice == "1":
            print_states_and_actions()

        elif choice == "2":
            print_transition_model()
            print_reward_examples()

        elif choice == "3":
            print_title("Esecuzione della Value Iteration")

            print("L'algoritmo aggiorna iterativamente il valore di ogni stato.")
            print("A ogni iterazione applica l'equazione di Bellman e misura la variazione delta.")
            print(f"L'esecuzione termina quando delta diventa minore di theta, pari a {THETA}.")

            values, policy, iterations, delta_history = value_iteration()

            print()
            print("Value Iteration completata.")
            print(f"Iterazioni eseguite: {iterations}")
            print(f"Delta finale: {delta_history[-1]:.6f}")

        elif choice == "4":
            values, policy, iterations, delta_history = compute_if_needed(
                values, policy, iterations, delta_history
            )
            print_value_function(values)

        elif choice == "5":
            values, policy, iterations, delta_history = compute_if_needed(
                values, policy, iterations, delta_history
            )
            print_policy(policy)

        elif choice == "6":
            values, policy, iterations, delta_history = compute_if_needed(
                values, policy, iterations, delta_history
            )
            print_action_values(values)

        elif choice == "7":
            values, policy, iterations, delta_history = compute_if_needed(
                values, policy, iterations, delta_history
            )
            print_convergence(delta_history)

        elif choice == "8":
            values, policy, iterations, delta_history = compute_if_needed(
                values, policy, iterations, delta_history
            )
            simulate_optimal_policy(policy)

        elif choice == "9":
            values, policy, iterations, delta_history = compute_if_needed(
                values, policy, iterations, delta_history
            )
            manual_mode(policy)

        elif choice == "10":
            values, policy, iterations, delta_history = compute_if_needed(
                values, policy, iterations, delta_history
            )
            save_report(values, policy, iterations, delta_history)

        elif choice == "0":
            print()
            print("Programma terminato.")
            break

        else:
            print()
            print("Scelta non valida. Riprova inserendo un numero presente nel menu.")


if __name__ == "__main__":
    main()