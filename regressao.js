    let model;
    async function trainModel() {
      const xs = tf.tensor2d([1,2,3,4,5],[5,1]);
      const ys = tf.tensor2d([5,7,9,11,13],[5,1]); // y = 2x + 3
      model = tf.sequential();
      model.add(tf.layers.dense({inputShape:[1], units:1}));
      model.compile({optimizer: tf.train.sgd(0.01), loss:'meanSquaredError'});
      await model.fit(xs, ys, {epochs:200});
      document.getElementById('regOutput').innerText = '✅ Modelo treinado! Digite um valor.';
    }

    async function predict() {
      const x = parseFloat(document.getElementById('xValue').value);
      if (isNaN(x)) return alert('Digite um número válido!');
      const pred = model.predict(tf.tensor2d([x],[1,1]));
      const y = (await pred.array())[0][0];
      document.getElementById('regOutput').innerText = `🔹 Previsão para x=${x}: y ≈ ${y.toFixed(2)}`;
    }

    trainModel();