# Lux AI Season 3 - NeurIPS 2024 Competition

## MLDS Society Team

### Overview
This repository contains our submission for the **Lux AI Season 3** competition, hosted as part of **NeurIPS 2024** on Kaggle. Our approach leverages **autoencoders (AE), NEAT (NeuroEvolution of Augmenting Topologies), and a game search strategy (GOM)** to develop an AI agent that optimally plays the game.

---

## Project Pipeline

### Input Data
- The agent receives **game state representations** consisting of **unit and resource positions**.
- This information is transformed using a **Game Object Model (GOM)**.

### Autoencoder (AE)
- The **GOM representations** are processed using an **autoencoder (AE)** to extract **latent embeddings**.
- The latent space representation is optimized for dimensionality reduction and feature extraction.

### NEAT Algorithm
- The **latent space** is used as input for a **NEAT-based AI agent**.
- The **NEAT algorithm** evolves an optimal strategy through reinforcement learning.

### User and Training
- The model undergoes **training with search-based game strategies** to enhance decision-making.
- A **user-controlled component** is integrated to allow for human-in-the-loop fine-tuning and evaluation.

---

## Technical Details
- **Positional Encoding:** Used for flattening and structuring latent embeddings.
- **Neuroevolution:** The NEAT algorithm evolves **neural architectures** to optimize performance.
- **Training Strategy:** Includes reinforcement learning and genetic search mechanisms.

## Contributors
- Ahmad Abdelhakam Mahmoud, President of the **Machine Learning and Data Science Society** of Manchester (MLDS)
- Samkitt Patni, member of MLDS
- Diyaco Shwany, member of MLDS
- Herbie Warner, member of MLDS
- Rithichan Chhorn, member of MLDS

For any inquiries, open an issue or reach out to the team!

---

## Links
- [Lux AI Competition](https://www.kaggle.com/c/lux-ai-season-3)
- [NeurIPS 2024](https://neurips.cc/)
- [NEAT Algorithm](https://en.wikipedia.org/wiki/Neuroevolution_of_augmenting_topologies)

