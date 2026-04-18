#!/bin/bash
echo "========================================"
echo "  Verificando dependencias del sistema  "
echo "========================================"

# Docker
if ! command -v docker &> /dev/null; then
  echo "❌ Docker NO está instalado. Descárgalo en: https://docs.docker.com/get-docker/"
  exit 1
else
  echo "✅ Docker: $(docker --version)"
fi

# Docker Compose
if ! docker compose version &> /dev/null; then
  echo "❌ Docker Compose NO está disponible."
  exit 1
else
  echo "✅ Docker Compose: $(docker compose version)"
fi

echo ""
echo "========================================"
echo "  Levantando servicios...               "
echo "========================================"
docker compose up --build -d

echo ""
echo "Esperando que la API inicie (10 seg)..."
sleep 10

echo ""
echo "========================================"
echo "  Probando endpoint /health             "
echo "========================================"
curl -s http://localhost:8000/health
echo ""
echo ""
echo "✅ ¡Todo listo! Abre http://localhost:8000/docs para probar la API."
echo "   n8n disponible en: http://localhost:5678 (usuario: admin / contraseña: admin123)"
