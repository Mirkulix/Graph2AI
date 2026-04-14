#!/bin/bash
set -e
echo "Installing QO systemd service..."
sudo cp scripts/qo.service /etc/systemd/system/qo.service
sudo systemctl daemon-reload
sudo systemctl enable qo
echo "Service installed. Start with: sudo systemctl start qo"
echo "Logs: journalctl -u qo -f"
