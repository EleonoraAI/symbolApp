def description(algorithm_type):
    if algorithm_type == "SIFT":
        desc = '''SIFT (Scale-Invariant Feature Transform) è un algoritmo progettato per estrarre caratteristiche chiave da un'immagine, che sono invarianti rispetto a scala e orientamento. Queste caratteristiche sono robuste rispetto a cambiamenti di scala, rotazione, illuminazione e sfondo.
        Funzionamento: SIFT individua punti chiave nell'immagine (chiamati keypoints) in vari livelli di scala. Per ogni punto chiave, calcola un descrittore basato sulle intensità dei pixel nelle vicinanze. Questi descrittori possono essere utilizzati per confrontare e abbinare punti chiave tra diverse immagini.'''
    elif algorithm_type == "SIFT_Canny":
        desc = '''SIFT (Scale-Invariant Feature Transform) è un algoritmo progettato per estrarre caratteristiche chiave da un'immagine, che sono invarianti rispetto a scala e orientamento. Queste caratteristiche sono robuste rispetto a cambiamenti di scala, rotazione, illuminazione e sfondo.'''
    elif algorithm_type == "Canny":
        desc = '''Il rilevatore di bordi di Canny è un algoritmo ampiamente utilizzato per individuare i contorni nelle immagini. È progettato per essere sensibile ai cambiamenti di intensità e allo stesso tempo ridurre al minimo le risposte fittizie dovute al rumore.
        Funzionamento: L'algoritmo di Canny utilizza un operatore di gradiente per individuare la variazione di intensità nell'immagine. Successivamente, applica la soppressione dei non massimi per sottolineare solo i punti in cui il gradiente è massimo. Infine, utilizza la tracciatura dei bordi per collegare i pixel di contorno in curve continue.'''
    elif algorithm_type == "HOG":
        desc = '''HOG (Histogram of Oriented Gradients) è un algoritmo utilizzato per l'estrazione di caratteristiche e il rilevamento di contorni. È ampiamente utilizzato nell'elaborazione delle immagini per il rilevamento di oggetti.
        Funzionamento: HOG calcola il gradiente dell'immagine per individuare i contorni. Quindi, calcola l'orientamento del gradiente per ogni pixel. Successivamente, crea un istogramma di orientamento dei gradienti per l'immagine. Infine, normalizza l'istogramma per ridurre l'effetto della luminosità.'''
    elif algorithm_type == "LBP":
        desc = '''LBP (Local Binary Pattern) è un algoritmo utilizzato per l'estrazione di caratteristiche e il rilevamento di contorni. È ampiamente utilizzato nell'elaborazione delle immagini per il rilevamento di oggetti.'''
    elif algorithm_type == "Sobel":
        desc = '''L'operatore di Sobel è un operatore di gradiente utilizzato per l'estrazione di caratteristiche e il rilevamento di contorni. È ampiamente utilizzato nell'elaborazione delle immagini per il rilevamento di oggetti.
        Funzionamento: L'operatore di Sobel utilizza due kernel separati per calcolare il gradiente approssimato per ogni punto dell'immagine. Uno dei kernel è utilizzato per calcolare le derivate parziali sull'asse X e l'altro sull'asse Y. Successivamente, calcola la magnitudine del gradiente per ogni pixel.'''
    elif algorithm_type == "Laplacian":
        desc = '''L'operatore di Laplacian è un operatore di gradiente utilizzato per l'estrazione di caratteristiche e il rilevamento di contorni. È ampiamente utilizzato nell'elaborazione delle immagini per il rilevamento di oggetti.
        Funzionamento: L'operatore di Laplacian calcola la seconda derivata di ogni pixel. Successivamente, calcola la magnitudine del gradiente per ogni pixel.'''
    elif algorithm_type == "FAST":
        desc = '''FAST (Features from Accelerated Segment Test) è un algoritmo utilizzato per l'estrazione di caratteristiche e il rilevamento di contorni. È ampiamente utilizzato nell'elaborazione delle immagini per il rilevamento di oggetti.'''
    elif algorithm_type == "Scharr":
        desc = '''L'operatore di Scharr è un operatore di gradiente utilizzato per l'estrazione di caratteristiche e il rilevamento di contorni. È ampiamente utilizzato nell'elaborazione delle immagini per il rilevamento di oggetti.
        Funzionamento: L'operatore di Scharr utilizza due kernel separati per calcolare il gradiente approssimato per ogni punto dell'immagine. Uno dei kernel è utilizzato per calcolare le derivate parziali sull'asse X e l'altro sull'asse Y. Successivamente, calcola la magnitudine del gradiente per ogni pixel.'''

    else:
        desc = "Algoritmo non riconosciuto"

    return desc