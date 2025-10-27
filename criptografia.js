    const ALFABETO = 'abcdefghijklmnopqrstuvwxyz ';
    let cryptoModel;

    async function trainCrypto() {
      const shift = 3;
      const X = [], Y = [];
      for (let c of ALFABETO) {
        let enc = ALFABETO[(ALFABETO.indexOf(c) + shift) % ALFABETO.length];
        X.push(ALFABETO.indexOf(c));
        Y.push(ALFABETO.indexOf(enc));
      }

      const xs = tf.oneHot(tf.tensor1d(X, 'int32'), ALFABETO.length);
      const ys = tf.oneHot(tf.tensor1d(Y, 'int32'), ALFABETO.length);

      cryptoModel = tf.sequential();
      cryptoModel.add(tf.layers.dense({inputShape:[ALFABETO.length], units:32, activation:'relu'}));
      cryptoModel.add(tf.layers.dense({units:ALFABETO.length, activation:'softmax'}));
      cryptoModel.compile({optimizer:'adam', loss:'categoricalCrossentropy'});

      await cryptoModel.fit(xs, ys, {epochs:100, verbose:0});
      document.getElementById('cryptoOutput').innerText = '✅ Modelo de criptografia treinado! Digite um texto.';
    }

    async function encrypt() {
      const text = (document.getElementById('plainText').value || '').toLowerCase();
      if (!text) return alert('Digite um texto!');
      if (!cryptoModel) await trainCrypto();

      const chars = text.split('');
      const validChars = chars.filter(c => ALFABETO.includes(c));
      if (validChars.length === 0) return alert('Use apenas letras minúsculas e espaço!');

      const oneHots = tf.oneHot(tf.tensor1d(validChars.map(c => ALFABETO.indexOf(c)), 'int32'), ALFABETO.length);
      const preds = cryptoModel.predict(oneHots);
      const resultIdx = await preds.argMax(1).data();
      const encrypted = Array.from(resultIdx).map(i => ALFABETO[i]).join('');
      document.getElementById('cryptoOutput').innerText = `🔒 Texto criptografado: ${encrypted}`;
    }