# Compte-rendu — Convolution pyramidale HEALPix avec NaN et jauges

**Date** : 15/09/2026
**Dépôt** : `GRID4EARTH/healpix-analyse` — fichiers déposés localement sur la machine de l'utilisateur (`C:\Users\jmdeloui\Documents\Claude\Projects\healpix-analyse`) via le pont device bridge, **sans commit ni push git**, conformément à la préférence de l'utilisateur.

---

## 1. Architecture retenue

L'implémentation s'appuie entièrement sur la pile existante du dépôt plutôt que
sur une nouvelle géométrie :

- **`HealPixDecomp`** (`healpix_analyse/decomp.py`, existant) fournit la
  pyramide de Laplace à reconstruction exacte (`S · W = I`), construite comme
  une chaîne de `HealPixDown`/`HealPixUp` appariés. C'est l'opérateur
  d'analyse/synthèse `W`/`S` utilisé partout dans cette mission.
- **`HealPixConv`** (`healpix_analyse/convol.py`, existant) fournit le noyau
  compact gauge-équivariant : un gabarit fixe de `P = kernel_sz²` points au
  pôle nord, transporté par jauge à chaque pixel cible, les données étant
  liées au gabarit par interpolation bilinéaire.

Trois nouveaux modules assemblent ces deux briques :

| Module | Rôle |
|---|---|
| `healpix_analyse/decomp.py` (étendu) | `HealPixDecomp.compute_weighted`/`invert` : gestion des NaN et poids de confiance |
| `healpix_analyse/kernel_pyramid.py` (nouveau) | `HealPixKernelPyramid` : un noyau `HealPixConv` fixe (non appris) par bande |
| `healpix_analyse/pyramid_conv.py` (nouveau) | `HealPixPyramidConv` : assemble decomp + kernel pyramid, avec la sémantique de division unique après synthèse |
| `healpix_analyse/validation.py` (nouveau) | Oracle de convolution directe, indépendant, pour valider/calibrer |

Aucune nouvelle géométrie n'a été introduite : les points d'évaluation du
noyau à chaque bande sont obtenus en réutilisant directement
`_local_kernel_grid` (la fonction privée qui construit le gabarit de
`HealPixConv`), pour garantir que le noyau pyramidal est évalué exactement
aux mêmes positions que le gabarit utilisé au moment de l'exécution.

Par défaut, toute la géométrie de ce travail utilise `ellipsoid="sphere"`
(et non `"WGS84"`, valeur par défaut historique du reste du dépôt) — voir
§6.

---

## 2. Décomposition du noyau et traitement inter-bandes

**Point mathématique central, explicité dans la documentation** (voir
`docs/pyramid_convolution.md`, section A.1) :

Soit `W`/`S` les opérateurs d'analyse/synthèse de `HealPixDecomp` (avec
`S·W = I` exactement). Le pipeline visé est `y = S·B·W·x`, qui n'égale la
vraie convolution `y = K·x` que pour `B = W·K·S` — un opérateur **dense
entre bandes** en général. **La décomposition n'est pas une
diagonalisation** : rien dans `S·W = I` n'implique qu'un `B` bloc-diagonal
(un noyau compact indépendant par bande, sans terme croisé) approxime bien
`K·x`.

`HealPixKernelPyramid` construit précisément cette approximation
**bloc-diagonale** : un `HealPixConv` fixe par bande, sans aucun couplage
inter-bandes. C'est un choix délibéré et documenté comme tel, pas une
propriété démontrée. Son erreur d'approximation est mesurée, pas supposée
(§5).

**Calibration (optionnelle)** : `HealPixKernelPyramid.calibrate` ajuste les
`P` poids d'une bande par moindres carrés contre un opérateur de référence,
**bande par bande** (pas d'optimisation jointe inter-bandes). Un bogue de
construction de la matrice de conception a été détecté et corrigé pendant
le développement (voir §7) — sans ce correctif, l'ajustement produisait des
noyaux incorrects avec une erreur ~4× plus grande que l'évaluation
analytique directe, y compris sur une cible pourtant représentable par une
seule bande.

Deux résultats honnêtes issus des tests (§5) :

- Sur une cible de même échelle que le gabarit (ex. gaussienne σ=1.2 px
  contre un gabarit 5×5), la calibration retrouve une précision quasi
  identique à l'évaluation analytique directe (≈5.1 % contre ≈4.8 % d'erreur
  RMS relative).
- Sur une cible bien plus large que le gabarit d'une seule bande (ex.
  σ=6 px), la calibration **ne comble pas l'écart** (≈76 % d'erreur RMS
  résiduelle) : reproduire une cible large par une cascade de noyaux
  compacts nécessite une optimisation **jointe** multi-niveaux — c'est
  précisément la contribution technique de l'article cité (Farbman, Fattal,
  Lischinski, *Convolution Pyramids*, ACM TOG 2011), **non implémentée
  ici**. Cette limite est documentée explicitement, sans reformulation qui
  la masquerait.

---

## 3. Traitement des NaN et des poids

`HealPixDecomp.compute_weighted(x, weights=None)` fait passer **le même**
opérateur linéaire `W` sur deux canaux :

```
q = W(m ⊙ x_safe)     # canal donnée, valeurs manquantes mises à 0
m̃ = W(m)              # canal poids, même opérateur, sur les poids seuls
```

Un noyau pyramidal `B_K` (les `HealPixConv` par bande) est appliqué **à
l'identique** aux deux canaux (`HealPixPyramidConv.apply_pyramid` ne permet
pas que `q` et `m̃` voient des noyaux différents), puis combiné par **une
seule division après synthèse** :

```
y = S(B_K q) / S(B_K m̃)
```

jamais bande par bande. Les bandes de détail de `m̃` sont des **termes de
correction signés**, pas des confiances par bande dans `[0, 1]` ; elles ne
sont ni seuillées ni divisées individuellement — seule la synthèse complète
`S(B_K m̃)` est une carte de support/confiance interprétable.

Cette propriété a été vérifiée exactement (à la précision machine, ≤1e-9 en
`float64`) : pour un champ constant `c` derrière un masque arbitraire
(y compris un grand trou contigu), `q = c·m̃` bande par bande, et
`HealPixPyramidConv` reconstruit exactement `c` partout où il y a du
support, `NaN` (ou `0`) exactement là où il n'y en a pas.

Le gradient traverse l'ensemble du pipeline `compute_weighted → noyau →
invert` ; la décision de masquage (`isfinite`) ne porte elle-même aucun
gradient (comme tout masquage booléen), mais les valeurs finies survivantes
restent pleinement différentiables — vérifié explicitement, y compris que
le gradient en un pixel toujours masqué est exactement nul.

---

## 4. Précision mesurée du gabarit 5×5, et quand en utiliser plus

**Sur champ spatialement lisse** (fonction sphérique basse fréquence),
comparé à un oracle de convolution directe indépendant (voir §7) : erreur
RMS relative ≈ 4–5 % pour un gabarit 5×5, pour les profils gaussien,
exponentiel, lorentzien et beta testés — nettement en dessous du seuil de
15 % fixé dans les tests.

**Sur bruit blanc** (contenu à l'échelle du pixel) : l'écart est bien plus
important — **≈22 % RMS mesuré** pour une gaussienne σ=1.2 px avec un
gabarit 5×5. Ce n'est pas un bogue mais une propriété réelle et mesurée de
`HealPixConv` lui-même : le gabarit fixe lie les données par **interpolation
bilinéaire** à des positions tournées qui ne coïncident presque jamais
exactement avec les centres des pixels voisins, ce qui approche bien
l'action d'un noyau continu sur un contenu lisse mais s'écarte nettement
d'une référence en quadrature au pixel près sur du contenu haute fréquence.
**Implication pratique** : cette convolution doit être considérée précise
pour des champs à bande limitée / lisses (quelques % RMS) ; ne pas
extrapoler cette précision à des affirmations par pixel sur des données
bruitées sans le vérifier sur des données de rugosité comparable.

**5×5 contre 7×7** (lorentzienne à queue large, σ=2.0 px, champ lisse) :
5×5 → 18.4 % RMS, 7×7 → 15.4 % RMS — une amélioration réelle mais modeste,
cohérente avec le fait qu'un noyau à queue lente bénéficie d'un support
plus large sans que 5×5 soit pour autant disqualifiant.

**Recommandation** : 5×5 convient pour des noyaux compacts à décroissance
rapide (gaussienne, exponentielle) sur des champs raisonnablement lisses ;
passer à 7×7 (ou plus) pour des noyaux à queue lourde (lorentzienne, beta à
faible β), et systématiquement mesurer — via
`healpix_analyse.validation.direct_spherical_convolution` — plutôt que
supposer, en particulier avant toute utilisation sur un champ dominé par du
bruit à l'échelle du pixel.

---

## 5. Tests et bancs de mesure exécutés, et sur quel matériel

Tous les tests ci-dessous ont été **exécutés réellement** dans un
bac à sable Linux cloud (aucun GPU disponible — voir §8), avec la pile
installée depuis PyPI (`torch 2.14.0+cu130` en mode CPU, `numpy 2.4.4`,
`healpix-geo`, `pyproj`, `scipy 1.17.1`) et le paquet `healpix-analyse`
installé en mode développement (`pip install -e .`) depuis un clone public
du dépôt.

- `tests/test_decomp_weighted.py` — 11 tests : reconstruction exacte
  inchangée, identité `q = c·m̃` pour un champ constant masqué (trou
  contigu inclus), reconstruction exacte du champ constant où il y a du
  support, poids explicites et poids nul = donnée manquante, carte
  totalement masquée → `NaN`/`0` selon `restore_mask`, gradient (y compris
  gradient nul sur pixel masqué), erreur de forme sur les poids, ciel
  partiel. **11/11 passés.**
- `tests/test_kernel_pyramid.py` — 15 tests : fidélité par bande contre
  l'oracle direct (champ lisse et bruit blanc, 4 familles de noyaux, tailles
  3×3/5×5/7×7), cohérence géométrique bande/decomp, exécution sur pyramide
  complète, calibration (cible représentable et cible large), noyau
  anisotrope. **15/15 passés**, avec les nombres mesurés ci-dessus imprimés
  explicitement dans la sortie de test (pas de nombre inventé).
- `tests/test_pyramid_conv.py` — 8 tests : préservation du champ constant
  (avec et sans masque), zone totalement masquée → `NaN`, trous aléatoires,
  poids explicites, mode `"signed"` (noyau signé, erreurs correctement
  levées si `weights`/`return_support` sont utilisés à tort), gradient de
  bout en bout, cohérence `apply_pyramid` + `invert` manuel. **8/8 passés.**
- Suite complète du dépôt (hors modules nécessitant `healpy`, absent de ce
  bac à sable — limitation d'environnement préexistante, sans rapport avec
  cette mission) : **322 tests passés**, 19 ignorés (`skip` préexistants),
  aucune régression introduite.

**Banc de mesure CPU** (`scripts/benchmark_pyramid_conv.py`, exécuté
réellement, 2 cœurs physiques, `float32`, gabarit 5×5, gaussien σ=1.2 px,
passe complète `compute_weighted → noyau → invert`) :

| level | npix | temps construction (s, une fois) | temps forward (ms) | forward/pixel (×1e-6 ms) |
|---:|---:|---:|---:|---:|
| 5 | 12 288 | 0.02 | 33.5 | 2.73 |
| 6 | 49 152 | 8.08 | 122.9 | 2.50 |
| 7 | 196 608 | 27.1 | 492.4 | 2.50 |

Le coût par pixel quasi constant sur une multiplication par 16 du nombre de
pixels est cohérent avec la complexité `O(N)` visée. **Aucun GPU n'était
disponible dans cet environnement** ; aucun chiffre GPU n'est rapporté, ni
extrapolé. Le coût de construction de la géométrie (mis en cache sur disque
par `HealPixConv` après le premier calcul à une configuration donnée) est
non négligeable à `level=6,7` et doit être amorti sur de nombreux appels,
pas répété à chaque appel.

---

## 6. Limites ouvertes, énoncées sans les masquer

1. **Bloc-diagonal uniquement** : aucun couplage inter-bandes n'est modélisé
   ni corrigé (§2).
2. **`calibrate` est bande-par-bande, pas jointe** : elle ne peut pas faire
   reproduire une cible large par un noyau compact d'une seule bande (§2).
   Une calibration jointe multi-niveaux (la contribution réelle de l'article
   cité) n'est pas implémentée.
3. **Pas d'oracle de validation automatisé pour l'anisotropie** : l'oracle
   indépendant (`healpix_analyse.validation`) ne couvre que les noyaux
   isotropes (documenté explicitement dans son docstring de module) ; les
   noyaux anisotropes ne sont vérifiés que qualitativement.
4. **Mode `"normalized"` avec noyau signé non pleinement caractérisé** :
   rien n'empêche un poids synthétisé négatif, qui n'est plus une confiance
   interprétable ; utiliser `mode="signed"` pour un noyau signé sur données
   entièrement finies.
5. **Cohérence de l'ellipsoïde non vérifiée automatiquement** : ce module
   utilise `ellipsoid="sphere"` par défaut partout, mais rien ne détecte
   automatiquement un `HealPixDecomp` construit avec `"WGS84"` (valeur par
   défaut historique ailleurs dans le dépôt) utilisé avec ce module.
6. **Pas de traitement mémoire par blocs/streaming au-delà du traitement
   bande par bande** : chaque bande est traitée comme un unique tenseur
   dense.
7. **Aucun chiffre GPU** (§5), faute de matériel disponible.
8. **`calibrate` est relativement lent** (boucle Python sur les `P` poids ×
   `n_excitations` par bande) — adapté à un usage hors-ligne, pas à un
   chemin critique.
9. **Pas de fusion/reformatage complet de la doc existante** : la
   documentation détaillée du présent travail est consolidée dans un seul
   nouveau fichier (`docs/pyramid_convolution.md`) plutôt que répartie sur
   les nombreux fichiers séparés suggérés en détail par la mission — un
   choix délibéré pour rester exhaustif sans fragmenter excessivement, à
   ajuster si une structure multi-fichiers est préférée.

---

## 7. Erreurs rencontrées et corrigées pendant le développement

Par souci de traçabilité, deux erreurs réelles ont été détectées et
corrigées avant livraison (aucune n'a été laissée sous le tapis) :

- **Confusion normalisée/brute** dans l'oracle de validation : la première
  version de `direct_spherical_convolution` divisait toujours par la somme
  des poids (convolution normalisée), alors que `HealPixConv` ne normalise
  jamais son noyau en interne (somme brute pondérée). Cela produisait des
  écarts de 400 % à l'évaluation initiale, purement dus à cette incohérence
  de convention — corrigé en ajoutant un paramètre `normalize` explicite et
  en comparant chaque fois la même convention des deux côtés.
- **Matrice de conception incorrecte dans `HealPixKernelPyramid.calibrate`**
  : la première version sondait chaque poids du noyau avec une impulsion de
  Dirac placée *au* pixel de sonde lui-même, ce qui ne correspond à la
  vraie contribution de ce poids que pour le point central du gabarit (les
  autres points, après rotation de jauge et interpolation bilinéaire,
  échantillonnent une position différente de celle du pixel de sonde). Cela
  produisait un ajustement incorrect (erreur ~4× plus grande que l'évaluation
  analytique, même sur une cible représentable). Corrigé en sondant avec un
  champ d'excitation aléatoire partagé et en lisant la réponse de chaque
  poids du noyau (noyau one-hot) au même ensemble de pixels de sonde, ce qui
  correspond exactement à la définition linéaire de l'opérateur.

---

## 8. Environnement d'exécution

- Sandbox cloud Linux, 2 cœurs CPU, pas de GPU.
- `torch 2.14.0+cu130` (mode CPU), `numpy 2.4.4`, `healpix-geo`, `pyproj`,
  `scipy 1.17.1`, `pytest`.
- Dépôt de départ : clone public `https://github.com/GRID4EARTH/healpix-analyse.git`
  (HEAD `bcdc907`), utilisé comme base de développement/test pour ne pas
  dépendre du pont vers la machine locale pendant tout le travail
  d'implémentation.
- **Livraison finale** : tous les fichiers nouveaux/modifiés ont été
  déposés sur la machine Windows de l'utilisateur
  (`C:\Users\jmdeloui\Documents\Claude\Projects\healpix-analyse`) via le
  pont `device bridge` (stage → commit, avec garde `expectedMtimeMs` sur
  chaque fichier préexistant). **Aucun `git commit` ni `git push` n'a été
  exécuté**, conformément à la préférence de l'utilisateur de maîtriser
  lui-même ses commits. Avant toute écriture, le contenu réellement présent
  sur la machine a été vérifié pour chaque fichier déjà modifié lors d'une
  phase antérieure (`docs/index.md`, `docs/overview.md`) afin de fusionner
  cette nouvelle mission sans écraser ces modifications antérieures ; les
  fichiers non touchés depuis (`decomp.py`, `__init__.py`, `docs/decomp.md`)
  ont été vérifiés identiques au clone de départ avant d'être réécrits.

---

## 9. Fichiers livrés

Nouveaux :
- `healpix_analyse/kernel_pyramid.py`
- `healpix_analyse/pyramid_conv.py`
- `healpix_analyse/validation.py`
- `docs/pyramid_convolution.md`
- `examples/pyramid_conv_quickstart.py`
- `scripts/benchmark_pyramid_conv.py`
- `tests/test_decomp_weighted.py`, `tests/test_kernel_pyramid.py`, `tests/test_pyramid_conv.py`

Modifiés :
- `healpix_analyse/decomp.py` (`compute_weighted`, `HealPixWeightedPyramid`, `invert` étendu)
- `healpix_analyse/__init__.py` (exports publics)
- `docs/decomp.md` (section NaN/poids)
- `docs/index.md`, `docs/overview.md` (référencement de la nouvelle page)

Aucune release, aucun merge, aucun commit git n'a été effectué — conforme à
la demande explicite de la mission et à la préférence connue de
l'utilisateur.
