// preprocess-worker.js
self.onmessage = async function(e) {
  // e.data: {imageBitmap, size}
  const { imageBitmap, size } = e.data;
  const canvas = new OffscreenCanvas(size, size);
  const ctx = canvas.getContext('2d');
  const [iw, ih] = [imageBitmap.width, imageBitmap.height];
  const r = Math.min(size / iw, size / ih);
  const newW = Math.round(iw * r), newH = Math.round(ih * r);
  const dx = Math.floor((size - newW) / 2), dy = Math.floor((size - newH) / 2);

  ctx.fillStyle = "#808080";
  ctx.fillRect(0, 0, size, size);
  ctx.drawImage(imageBitmap, dx, dy, newW, newH);

  const imageData = ctx.getImageData(0, 0, size, size).data;
  const floatArray = new Float32Array(size * size * 3);
  for (let i = 0; i < size * size; ++i) {
    floatArray[i] = imageData[i * 4] / 255.0;
    floatArray[size * size + i] = imageData[i * 4 + 1] / 255.0;
    floatArray[2 * size * size + i] = imageData[i * 4 + 2] / 255.0;
  }
  self.postMessage({ floatArray }, [floatArray.buffer]);
}
