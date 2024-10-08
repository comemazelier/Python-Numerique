---
jupytext:
  cell_metadata_json: true
  encoding: '# -*- coding: utf-8 -*-'
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
language_info:
  name: python
  nbconvert_exporter: python
  pygments_lexer: ipython3
nbhosting:
  title: suite du TP simple avec des images
---

Licence CC BY-NC-ND, Valérie Roy & Thierry Parmentelat

+++

# TP images (2/2)

merci à Wikipedia et à stackoverflow

**le but de ce TP n'est pas d'apprendre le traitement d'image  
on se sert d'images pour égayer des exercices avec `numpy`  
(et parce que quand on se trompe ça se voit)**

```{code-cell} ipython3
import numpy as np
import numpy.random as rd
from matplotlib import pyplot as plt
```

+++ {"tags": ["framed_cell"]}

````{admonition} → **notions intervenant dans ce TP**

* sur les tableaux `numpy.ndarray`
  * `reshape()`, masques booléens, *ufunc*, agrégation, opérations linéaires
  * pour l'exercice `patchwork`:  
    on peut le traiter sans, mais l'exercice se prête bien à l'utilisation d'une [indexation d'un tableau par un tableau - voyez par exemple ceci](https://ue12-p24-numerique.readthedocs.io/en/main/1-14-numpy-optional-indexing-nb.html)
  * pour l'exercice `sepia`:  
    ici aussi on peut le faire "naivement" mais l'utilisation de `np.dot()` peut rendre le code beaucoup plus court
* pour la lecture, l'écriture et l'affichage d'images
  * utilisez `plt.imread()`, `plt.imshow()`
  * utilisez `plt.show()` entre deux `plt.imshow()` si vous affichez plusieurs images dans une même cellule

  ```{admonition} **note à propos de l'affichage**
  :class: seealso dropdown admonition-small

  * nous utilisons les fonctions d'affichage d'images de `pyplot` par souci de simplicité
  * nous ne signifions pas là du tout que ce sont les meilleures!  
    par exemple `matplotlib.pyplot.imsave` ne vous permet pas de donner la qualité de la compression  
    alors que la fonction `save` de `PIL` le permet
  * vous êtes libres d'utiliser une autre librairie comme `opencv`  
    si vous la connaissez assez pour vous débrouiller (et l'installer), les images ne sont qu'un prétexte...
  ```
````

+++

## Création d'un patchwork

+++

1. Le fichier `data/rgb-codes.txt` contient une table de couleurs:
```
AliceBlue 240 248 255
AntiqueWhite 250 235 215
Aqua 0 255 255
.../...
YellowGreen 154 205 50
```
Le nom de la couleur est suivi des 3 valeurs de ses codes `R`, `G` et `B`  
Lisez cette table en `Python` et rangez-la dans la structure qui vous semble adéquate.

```{code-cell} ipython3
:scrolled: true

# votre code
filename = 'data/rgb-codes.txt'
colors = dict()
with open(filename, 'r') as file:
    for line in file:
        colname, *l = line.split()
        colors[colname] = np.array([int(elt) for elt in l])
colors
```

2. Affichez, à partir de votre structure, les valeurs rgb entières des couleurs suivantes  
`'Red'`, `'Lime'`, `'Blue'`

```{code-cell} ipython3
# votre code
print(colors['Red'], colors['Lime'], colors['Blue'])
```

3. Faites une fonction `patchwork` qui  

   * prend une liste de couleurs et la structure donnant le code des couleurs RGB
   * et retourne un tableau `numpy` avec un patchwork de ces couleurs  
   * (pas trop petits les patchs - on doit voir clairement les taches de couleurs  
   si besoin de compléter l'image mettez du blanc

+++

````{admonition} indices
:class: dropdown
  
* sont potentiellement utiles pour cet exo:
  * la fonction `np.indices()`
  * [l'indexation d'un tableau par un tableau](https://ue12-p24-numerique.readthedocs.io/en/main/1-14-numpy-optional-indexing-nb.html)
* aussi, ça peut être habile de couper le problème en deux, et de commencer par écrire une fonction `rectangle_size(n)` qui vous donne la taille du patchwork en fonction du nombre de couleurs  
  ```{admonition} et pour calculer la taille au plus juste
  :class: tip dropdown

  en version un peu brute, on pourrait utiliser juste la racine carrée;
  par exemple avec 5 couleurs créer un carré 3x3 - mais 3x2 c'est quand même mieux !

  voici pour vous aider à calculer le rectangle qui contient n couleurs

  n | rect | n | rect | n | rect | n | rect |
  -|-|-|-|-|-|-|-|
  1 | 1x1 | 5 | 2x3 | 9 | 3x3 | 14 | 4x4 |
  2 | 1x2 | 6 | 2x3 | 10 | 3x4 | 15 | 4x4 |
  3 | 2x2 | 7 | 3x3 | 11 | 3x4 | 16 | 4x4 |
  4 | 2x2 | 8 | 3x3 | 12 | 3x4 | 17 | 4x5 |
  ```
````

```{code-cell} ipython3
# votre code
def rectangle_size(n):
    
    i = 2
    j = 2
    
    if n == 1 :
        return(1,1)
              
    elif n == 2 :
        return(2,1)
    
    else :
        while i*j < n :
            i += 1
            if i*j >= n :
                return(i,j)
            j += 1
            if i*j >= n :
                return(i,j)
        return(i,j)

def patchwork(liste_couleurs, code_rgb):
    #on suppose que code_rgb est un np.array qui contient les codes rgb des couleurs de la liste, dans l'ordre
    n = len(liste_couleurs)
    i,j = rectangle_size(n)

    #on rajoute la couleur blanche à la fin de code_rgb
    blanc = np.array([[255,255,255]])
    nouveau_code_rgb = np.concatenate((code_rgb, blanc), axis = 0)

    #on va compléter les cases vides en blanc
    pattern_initial = np.arange(n)
    completer_en_blanc = np.array( ((i*j) - n) * [n])
    pattern = np.concatenate((pattern_initial, completer_en_blanc), axis = None)
    

    # on donne à pattern la forme voulue, et on s'assure qu'il est bien au format uint8
    pattern = pattern.reshape(i,j)
    nouveau_pattern = np.astype(pattern, 'uint8')
    return (np.astype(nouveau_code_rgb[nouveau_pattern], 'uint8'))


    
```

4. Tirez aléatoirement une liste de couleurs et appliquez votre fonction à ces couleurs.

```{code-cell} ipython3
# n est le nombre de couleurs qu'on veut avoir
n = 150

clefs = np.array(list(colors.keys()))
length = len(clefs)

# on définit la liste de couleurs aléatoires qu'on pioche 
# dans le document txt initial, et l'array code_rgb associé
liste_couleurs = [clefs[i] for i in rd.randint(0, length, n)]
code_rgb = np.array([colors[couleur] for couleur in liste_couleurs])

plt.imshow(patchwork(liste_couleurs, code_rgb))
```

5. Sélectionnez toutes les couleurs à base de blanc et affichez leur patchwork  
même chose pour des jaunes

```{code-cell} ipython3
# votre code
# On crée un masque qui donne la position des couleurs contenant 'White' dans leur nom
# On récupère ensuite la liste de ces couleurs
masque_liste_blancs = np.array(['White' in elt for elt in clefs])
liste_blancs = clefs[masque_liste_blancs]

# On crée l'array qui contient les valeurs rgb correspondantes
code_rgb_blancs = np.array([colors[couleur] for couleur in liste_blancs])

plt.imshow(patchwork(liste_blancs, code_rgb_blancs))
```

```{code-cell} ipython3
:scrolled: true

masque_liste_jaunes = np.array(['Yellow' in elt for elt in clefs])
liste_jaunes = clefs[masque_liste_jaunes]

code_rgb_jaunes = np.array([colors[couleur] for couleur in liste_jaunes])

plt.imshow(patchwork(liste_jaunes, code_rgb_jaunes))
```

6. Appliquez la fonction à toutes les couleurs du fichier  
et sauver ce patchwork dans le fichier `patchwork.png` avec `plt.imsave`

```{code-cell} ipython3
# votre code
# on récupère l'array de tous les rgb disponibles
# clefs est la liste des clefs du dictionnaire colors (définie plus haut)
code_rgb_total = np.array([colors[nom] for nom in clefs])

patch = patchwork(clefs, code_rgb_total)
plt.imsave('patchwork.png', patch)
```

7. Relisez et affichez votre fichier  
   attention si votre image vous semble floue c'est juste que l'affichage grossit vos pixels

vous devriez obtenir quelque chose comme ceci

```{image} media/patchwork-all.jpg
:align: center
```

```{code-cell} ipython3
# votre code
im = plt.imread('patchwork.png')
plt.imshow(im)
```

## Somme dans une image & overflow

+++

0. Lisez l'image `data/les-mines.jpg`

```{code-cell} ipython3
# votre code
im = plt.imread('data/les-mines.jpg')
```

1. Créez un nouveau tableau `numpy.ndarray` en sommant **avec l'opérateur `+`** les valeurs RGB des pixels de votre image

```{code-cell} ipython3
# votre code
# on procède par slicing
image_grise = im[:,:,0]+im[:,:,1]+im[:,:,2]
```

2. Regardez le type de cette image-somme, et son maximum; que remarquez-vous?  
   Affichez cette image-somme; comme elle ne contient qu'un canal il est habile de l'afficher en "niveaux de gris" (normalement le résultat n'est pas terrible ...)


   ```{admonition} niveaux de gris ?
   :class: dropdown tip

   cherchez sur google `pyplot imshow cmap gray`
   ```

```{code-cell} ipython3
# votre code
print(image_grise.dtype,image_grise.max())
plt.imshow(image_grise, cmap = 'grey')
#on remarque que le maximum est 255, car le résultat doit encore etre de type uint8 ?
```

3. Créez un nouveau tableau `numpy.ndarray` en sommant mais cette fois **avec la fonction d'agrégation `np.sum`** les valeurs RGB des pixels de votre image

```{code-cell} ipython3
# votre code
im_gris = np.sum(im, axis = 2)
```

4. Comme dans le 2., regardez son maximum et son type, et affichez la

```{code-cell} ipython3
# votre code
print(im_gris.dtype,im_gris.max())
plt.imshow(im_gris, cmap = 'grey')
```

5. Les deux images sont de qualité très différente, pourquoi cette différence ? Utilisez le help `np.sum?`

```{code-cell} ipython3
# votre code / explication
# La différence de qualité vient du type de nombre renvoyé par les deux sommes. 
# Avec '+', on reste en uint8, tandis qu'avec np.sum on obtient des uint64
```

6. Passez l'image en niveaux de gris de type entiers non-signés 8 bits  
(de la manière que vous préférez)

```{code-cell} ipython3
# votre code
im_gris = np.astype(im_gris, 'uint8')
```

7. Remplacez dans l'image en niveaux de gris,  
les valeurs >= à 127 par 255 et celles inférieures par 0  
Affichez l'image avec une carte des couleurs des niveaux de gris  
vous pouvez utilisez la fonction `numpy.where`

```{code-cell} ipython3
# votre code
im_gris[im_gris >= 127] = 255
im_gris[im_gris < 127] = 0
plt.imshow(im_gris, cmap = 'grey')
```

8. avec la fonction `numpy.unique`  
regardez les valeurs différentes que vous avez dans votre image en noir et blanc

```{code-cell} ipython3
# votre code
np.unique(im_gris)
```

## Image en sépia

+++

Pour passer en sépia les valeurs R, G et B d'un pixel  
(encodées ici sur un entier non-signé 8 bits)  

1. on transforme les valeurs `R`, `G` et `B` par la transformation  
`0.393 * R + 0.769 * G + 0.189 * B`  
`0.349 * R + 0.686 * G + 0.168 * B`  
`0.272 * R + 0.534 * G + 0.131 * B`  
(attention les calculs doivent se faire en flottants pas en uint8  
pour ne pas avoir, par exemple, 256 devenant 0)  
1. puis on seuille les valeurs qui sont plus grandes que `255` à `255`
1. naturellement l'image doit être ensuite remise dans un format correct  
(uint8 ou float entre 0 et 1)

+++

````{tip}
jetez un coup d'oeil à la fonction `np.dot` 
qui est si on veut une généralisation du produit matriciel

dont voici un exemple d'utilisation:
````

```{code-cell} ipython3
:scrolled: true

# exemple de produit de matrices avec `numpy.dot`
# le help(np.dot) dit: dot(A, B)[i,j,k,m] = sum(A[i,j,:] * B[k,:,m])

i, j, k, m, n = 2, 3, 4, 5, 6
A = np.arange(i*j*k).reshape(i, j, k)
B = np.arange(m*k*n).reshape(m, k, n)

C = A.dot(B)
# or C = np.dot(A, B)

print(f"en partant des dimensions {A.shape} et {B.shape}")
print(f"on obtient un résultat de dimension {C.shape}")
print(f"et le nombre de termes dans chaque `sum()` est {A.shape[-1]} == {B.shape[-2]}")
```

**Exercice**

+++

1. Faites une fonction qui prend en argument une image RGB et rend une image RGB sépia  
la fonction `numpy.dot` peut être utilisée si besoin, voir l'exemple ci-dessus

```{code-cell} ipython3
# votre code
# On définit d'abord la matrice qui apparait dans les opérations pour obtenir du sépia
matrice_sepia = np.array([[0.393,0.349,0.272],[0.769,0.686,0.534],[0.189,0.168,0.131]])

def sepia(image_rgb):
    
    # On effectue les opérations nécessaires
    image_de_transition = np.dot(image_rgb, matrice_sepia)
    
    # On fixe les valeurs trop élevées à 255
    image_de_transition[image_de_transition >= 255] = 255
    
    # On veut des uint8 donc on arrondit à l'entier le plus proche
    image_de_transition = np.round(image_de_transition)
    
    # On passe les valeurs de l'array en format uint8
    image_sepia = np.astype(image_de_transition, 'uint8')
    
    return image_sepia
```

2. Passez votre patchwork de couleurs en sépia  
Lisez le fichier `patchwork-all.jpg` si vous n'avez pas de fichier perso

```{code-cell} ipython3
# votre code
plt.imshow(sepia(patch))
```

3. Passez l'image `data/les-mines.jpg` en sépia

```{code-cell} ipython3
# votre code
im = plt.imread('data/les-mines.jpg')
plt.imshow(sepia(im))
```

## Exemple de qualité de compression

+++

1. Importez la librairie `Image`de `PIL` (pillow)  
(vous devez peut être installer PIL dans votre environnement)

```{code-cell} ipython3
# votre code
from PIL import Image
```

2. Quelle est la taille du fichier `data/les-mines.jpg` sur disque ?

```{code-cell} ipython3
file = "data/les-mines.jpg"
```

```{code-cell} ipython3
# votre code
im = plt.imread(file, 'uint8')
i,j,k = im.shape
print(f' {i*j*k} octets' )
```

3. Lisez le fichier 'data/les-mines.jpg' avec `Image.open` et avec `plt.imread`

```{code-cell} ipython3
# votre code
image_Image = Image.open(file)
image_plt = plt.imread(file)
```

4. Vérifiez que les valeurs contenues dans les deux objets sont proches

```{code-cell} ipython3
# votre code
print(image_Image - image_plt)
```

5. Sauvez (toujours avec de nouveaux noms de fichiers)  
l'image lue par `imread` avec `plt.imsave`  
l'image lue par `Image.open` avec `save` et une `quality=100`  
(`save` s'applique à l'objet créé par `Image.open`)

```{code-cell} ipython3
# votre code
plt.imsave('image_plt.png', image_plt)
image_Image.save('image_Image.png',quality=100)
```

6. Quelles sont les tailles de ces deux fichiers sur votre disque ?  
Que constatez-vous ?

```{code-cell} ipython3
# votre code
#Le fichier enregistré avec plt.imsave pèse 1,1 MB
#Le fichier enregistré avec save pèse 970 KB, il est donc plus léger que le précédent
```

7. Relisez les deux fichiers créés et affichez avec `plt.imshow` leur différence

```{code-cell} ipython3
# votre code
image_Image = Image.open(file)
image_plt = plt.imread(file)
plt.imshow(image_Image - image_plt)

# il n'y a donc pas de différence entre les deux fichiers, puis le noir est codé par 0,0,0 en rgb
```
