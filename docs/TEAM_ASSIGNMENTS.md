# ResQPet - Assegnazione Documentazione

## Branch per Membro

| Persona | Branch | File Assegnati |
|---------|--------|----------------|
| **P1** | `docs/p1-architettura-fusion` | `02_architettura.tex`, `08_fusion.tex` |
| **P2** | `docs/p2-backbone-collar` | `03_backbone.tex`, `04_collar.tex` |
| **P3** | `docs/p3-skin-pose` | `05_skin.tex`, `06_pose.tex` |
| **P4** | `docs/p4-breed-risultati` | `07_breed.tex`, `09_risultati.tex` |
| **P5** | `docs/p5-report-coordinator` | `report.tex`, `figures/` |

---

## Istruzioni per ogni Membro

### 1. Clona e vai sul tuo branch
```bash
git clone <repo-url>
cd ResQPet
git checkout docs/pX-nome-branch   # Sostituisci X con il tuo numero
```

### 2. Lavora SOLO sui tuoi file
```
docs/report/sections/XX_tuofile.tex
```

### 3. Commit e push delle modifiche
```bash
git add docs/report/sections/tuofile.tex
git commit -m "docs(sezione): descrizione modifiche"
git push origin docs/pX-nome-branch
```

### 4. Quando hai finito, apri una Pull Request
- Base: `main`
- Compare: `docs/pX-nome-branch`
- Titolo: `[DOCS] Sezione XYZ completata`

---

## Convenzione Commit Messages

```
docs(architettura): aggiunta descrizione pipeline
docs(backbone): fix formula YOLO loss
docs(fusion): tabella pesi ensemble
docs(risultati): grafici accuracy per modello
```

---

## Compilazione LaTeX (per test locale)

```bash
cd docs/report
pdflatex report.tex
bibtex report
pdflatex report.tex
pdflatex report.tex
```

Oppure usa **Overleaf** importando la cartella `docs/report/`.

---

## Deadline e Coordinamento

- [ ] P1 - Architettura + Fusion
- [ ] P2 - Backbone + Collar
- [ ] P3 - Skin + Pose
- [ ] P4 - Breed + Risultati
- [ ] P5 - Report main + Figure

**Coordinatore (P5)**: responsabile del merge finale e della coerenza del documento.
