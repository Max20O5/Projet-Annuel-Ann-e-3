#pragma once  // Toujours mettre ça en première ligne pour éviter les inclusions multiples
#include <vector>

class LinearModel {
private:
    std::vector<double> weights;
    double bias;

public:
    // 1. Le Constructeur : Créer le modèle
    // input_size : Nombre d'entrées (ex: 32*32*3 pour une image)
    LinearModel(int input_size);

    // 2. Prédiction : Estime une valeur (Forward pass)
    // "const ... &" signifie : "Je lis le vecteur sans le copier" (Optimisation Vitesse)
    double predict(const std::vector<double>& inputs) const;

    // 3. Entraînement : Corrige les poids (Backward pass)
    void train(const std::vector<double>& inputs, const std::vector<double>& labels, double learning_rate, int epochs);
    
    // 4. Sauvegarde
    void save(const char* filename);
};