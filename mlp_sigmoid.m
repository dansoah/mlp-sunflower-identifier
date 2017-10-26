%Executa uma rede do tipo MLP com uma camada oculta
function Y = mlp_sigmoid(Wo, Ws, teste)
 
    % Forward step for hidden layer
    Ao = [1 teste] * Wo;
    Yo = 1 ./ (1 + exp(-Ao));

    % Forward step for output layer
    As = [1 Yo] * Ws;
    Y = 1 ./ (1 + exp(-As));
end
