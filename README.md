# A2 — Reconhecimento Semântico em Publicidade Visual com CLIP

> Ler o conteúdo de **2.600+ anúncios e imagens** sem treinar nada: só os embeddings pré-treinados do CLIP,
> cosine similarity e linguagem natural. Rankeia os objetos mais frequentes da publicidade e busca imagens por frase.

<p>
<img alt="Python" src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white">
<img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-CPU%20%7C%20CUDA-EE4C2C?logo=pytorch&logoColor=white">
<img alt="CLIP" src="https://img.shields.io/badge/CLIP-ViT--B%2F32-5B21B6">
<img alt="Zero-shot" src="https://img.shields.io/badge/Zero--shot-sem%20treino-16A34A">
<img alt="License" src="https://img.shields.io/badge/License-MIT-blue">
</p>

Projeto da disciplina **Deep Learning & Computer Vision**. Entregável autossuficiente: um único notebook que roda do
início ao fim no **Google Colab (GPU T4)** sem edições.

---

## Destaques

- **Zero treino.** Nenhum peso é ajustado. Toda a "inteligência" vem do pré-treino contrastivo do CLIP sobre ~400M
  pares (imagem, legenda). O rótulo vira **qualquer texto**.
- **Ranking semântico** de 26 conceitos por frequência no corpus, com **threshold justificado por evidência**
  (z-score por conceito vs. global μ+kσ vs. softmax zero-shot).
- **Busca texto→imagem** com 10 consultas que vão do **genérico ao específico** e do **concreto ao abstrato** —
  incluindo os casos onde o CLIP interpreta a frase de um jeito inesperado.
- **Validação quantitativa**, não só visual: matriz **conceito × categoria** confirma se *"a car"* realmente se
  concentra em anúncios automotivos.
- **Análise teórica** do porquê o alinhamento visual-textual do CLIP habilita zero-shot, e uma comparação lado a
  lado da tokenização **CLIP vs. BERT** (padding, attention mask) com demonstração em código.

## O que o projeto faz

| Etapa | Entrada | Saída |
|---|---|---|
| **2.1 Ranking** | imagens do corpus × 26 descrições de conceito | conceitos mais frequentes + score + top-5 ilustrado |
| **2.2 Busca** | 10 consultas em linguagem natural | top-5 imagens por consulta + análise de acerto/erro |
| **Teoria** | — | alinhamento visual-textual; CLIP vs. BERT (tokenização) |

Pipeline: **varredura + dedupe MD5 → embeddings CLIP (L2-norm, com cache) → similaridade → threshold → ranking /
busca → visualização**.

## Dataset

**ADS-16** (Roffo & Vinciarelli, 2016) — Kaggle [`groffo/ads16-dataset`](https://www.kaggle.com/datasets/groffo/ads16-dataset).
Duas partes; cada uma contém:

- `Ads/Ads/<n>/` — **anúncios curados**, 20 categorias de produto (a pasta `<n>` mapeia para `Cat(n-1)`), ~15 imagens
  cada, **301 no total** (todas únicas por MD5).
- `Corpus/Corpus/U####/IM-POS` e `IM-NEG` — **imagens pessoais** que cada um dos 120 usuários marcou como
  preferida/rejeitada (5 + 5 por usuário), com legendas livres (ex. *"my cats"*).
- `Documents/` — apenas o PDF do artigo.

**Corpus de trabalho: o dataset inteiro (~2.650 imagens únicas após dedupe MD5).** Os 301 anúncios curados sozinhos
ficam abaixo do mínimo de 500 exigido; a rubrica aceita *"o corpus inteiro **ou** um subconjunto de ≥500"*, então uso
o corpus inteiro (anúncios + imagens de preferência). O rótulo de categoria de produto só existe nos 301 anúncios,
então a validação **conceito × categoria** é feita sobre esse subconjunto rotulado, enquanto ranking e busca rodam
sobre o corpus completo.

> **Categorias (20):** Clothing & Shoes · Automotive · Baby Products · Health & Beauty · Media (BMVD) ·
> Consumer Electronics · Console & Video Games · DIY & Tools · Garden & Outdoor living · Grocery · Kitchen & Home ·
> Betting · Jewellery & Watches · Musical Instruments · Office Products · Pet Supplies · Computer Software ·
> Sports & Outdoors · Toys & Games · Dating Sites.

Credencial: `~/.kaggle/kaggle.json` (Kaggle → Settings → Create New Token). **Nunca versionada** (está no `.gitignore`).

## Por que CLIP

O CLIP tem dois encoders — imagem (ViT) e texto (Transformer) — que projetam suas saídas num **espaço de embedding
comum**, treinado de forma **contrastiva (InfoNCE simétrica)** sobre ~400M pares (imagem, legenda) da web. Dentro de
cada batch, os pares corretos ficam na diagonal (positivos) e todo o resto são negativos; uma **temperatura
aprendível** escala as similaridades.

Esse alinhamento é o que habilita **recuperação e classificação zero-shot**: o "rótulo" deixa de ser uma classe fixa
e passa a ser qualquer frase, e a cosine similarity mede a correspondência semântica — sem nenhum treino
supervisionado sobre o ADS-16. É exatamente o que a atividade pede.

## Estrutura

```
A2_clip_ads16/
├── notebooks/
│   └── A2_clip_ads16.ipynb    # ENTREGÁVEL — roda sozinho no Colab T4
├── figures/                    # figuras geradas pelo notebook
├── data/                       # ADS-16 (gitignored)
├── cache/                      # embeddings .npy (gitignored)
├── requirements.txt            # versões fixadas
├── README.md
└── LICENSE
```

## Setup

```bash
python -m venv venv
source venv/Scripts/activate      # Windows Git Bash;  Linux/Mac: source venv/bin/activate
pip install -r requirements.txt
# credencial do Kaggle em ~/.kaggle/kaggle.json
```

## Rodar

**Colab (recomendado, GPU T4).** Abrir `notebooks/A2_clip_ads16.ipynb`, selecionar runtime **T4**, executar todas as
células. O notebook baixa o dataset via Kaggle API (pede o upload do `kaggle.json`) e roda do início ao fim sem
edições.

**Local (CPU).** Com o dataset já em `data/`, executar o notebook a partir de `notebooks/`. O CLIP ViT-B/32 embeda o
corpus em poucos minutos em CPU; os embeddings ficam em cache (`cache/`), então reexecuções são rápidas. A comparação
com o ViT-L/14 (§3.4) é pesada em CPU — desligar com `RUN_L14 = False`.

Validação não interativa:

```bash
jupyter nbconvert --to notebook --execute notebooks/A2_clip_ads16.ipynb
```

## Resultados

_Preencher após a execução (as figuras abaixo são geradas pelo notebook em `figures/`)._

| Ranking por frequência | Conceito × categoria |
|---|---|
| ![Ranking](figures/ranking_frequencia.png) | ![Heatmap](figures/conceito_x_categoria.png) |

| Top-5 dos conceitos mais frequentes |
|---|
| ![Top-5 conceitos](figures/top5_conceitos.png) |

| Exemplo de busca texto→imagem |
|---|
| ![Busca](figures/busca_1.png) |

## Status (checklist da rubrica)

- [ ] Pipeline CLIP sobre o corpus ADS-16 completo (≥500 imagens — justificado em §1.1)
- [ ] Cosine similarity de cada imagem com **26** descrições de conceitos (≥20)
- [ ] Ranking por frequência com score médio e **threshold justificado** (z-score por conceito vs. alternativas, §3.1)
- [ ] Top-5 conceitos visualizados com exemplos do corpus (§3.2)
- [ ] Concentração conceito × categoria como evidência quantitativa (§3.3)
- [ ] Busca texto→imagem, top-5, **10 consultas** (genérico→específico, concreto→abstrato) (§4)
- [ ] Análise por consulta: recuperou o esperado ou interpretou de forma inesperada? (§4.1)
- [ ] Análise escrita: alinhamento visual-textual e por que o pré-treino contrastivo habilita zero-shot (§5.1)
- [ ] Comparação CLIP vs. BERT: padding e attention mask, com demonstração em código (§5.2)

## Uso de IA

Desenvolvido com apoio de **Claude (Anthropic) via Claude Code**, conforme a política "Sinal Verde" da disciplina:
estruturação do projeto, esqueleto do notebook e revisão de código. Todo o código foi executado, revisado e validado
pelo autor; as análises foram verificadas contra os resultados reais.

## Referências

- Radford et al. (2021). *Learning Transferable Visual Models From Natural Language Supervision* (CLIP). arXiv:2103.00020.
- Roffo & Vinciarelli (2016). *Personality in Computational Advertising: A Benchmark* (ADS-16).
- Devlin et al. (2019). *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*. arXiv:1810.04805.

## Licença

MIT — ver [`LICENSE`](LICENSE).
