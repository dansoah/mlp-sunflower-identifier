%Implementação do back-propagation simples para MLP de três camadas
%As dimensões das matrizes de pesos devem ser compatíveis com as dimensões das matrizes de entrada e saidas esperadas
%A primeira linha de cada matriz de peso correspondem aos pesos de bias
%O número de entradas sendo usadas é dada pelo número de linhas de X e T (devem ser iguais)
%Os vetores são representados por matrizes-linha
function [Wo, Ws, errVec] = backprop_sigmoid(Dados, Alvos, nHidden, eta, nEpocas, minError)
  nClasses = columns(Alvos);
  nDim = columns(Dados);

%Wo = [ 0. 0.; 0.  0.;  0. 0. ];
Wo = 2*rand(nDim+1,nHidden) - 1.0; %Inicialização aleatória entre -1 e 1
%Ws = [ 0; 0; 0 ];
Ws = 2*rand(nHidden+1,nClasses) - 1.0 ; %Inicialização aleatória entre -1 e 1

%A plotagem do vetor de erros (errVec) mostra a curva de aprendizado
errVec = zeros(1,nEpocas);
%O número de épocas é dado pelo comprimento do vetor de erros
for epoch = 1:length(errVec)
  ordering = randperm(rows(Dados));
  X_ = Dados(ordering,:);
  T_ = Alvos(ordering,:);
  err2 = 0;
  for i = 1:rows(X_);
    % Forward step for hidden layer
    Ao = [1 X_(i,:)] * Wo;
    Yo = 1 ./ (1 + exp(-Ao));

    % Forward step for output layer
    As = [1 Yo] * Ws;
    Ys = 1 ./ (1 + exp(-As));

    % Backward step for output layer
    Es = [ T_(i,:) - Ys ];
    Ds = Es .* Ys.*(1-Ys);

    % Backward step for the hidden layer
    Eo = Ds*Ws(2:end,:)';
    Do_ = Eo .* Yo .* (1-Yo);

    % Weight update
    Ws = Ws + eta * [1 Yo]'*Ds;
    Wo = Wo + eta * [1 X_(i,:)]'*Do_;
    
    err2 += sum(Es.^ 2);
  end
  errVec(epoch) = err2/rows(Dados);
  if err2 < minError break; end
end