#!/bin/bash

echo "Uitgevoerd op $(date)" >> /home/projector/Documents/network_backup_log.txt

# Strikte foutafhandeling: stop bij fouten, ongedefinieerde variabelen of pipe-fouten
set -euo pipefail

# Controleer of het script als root (sudo) wordt uitgevoerd
if [ "$EUID" -ne 0 ]; then
  echo "Fout: Dit script moet met root-rechten worden uitgevoerd. Gebruik 'sudo ./setup_network.sh'"
  exit 1
fi

# Variabelen
WIFI_SSID="Tabloo Robot"
WIFI_PASS="fuj9akr.NVX*azb8cdn"
WIFI_IFACE="wlP1p1s0" # Pas aan indien je Orin Nano een andere interface naam gebruikt

# --- 0. Bestandsrechten herstellen ---
echo "Bestandsrechten van netwerkprofielen herstellen..."
chmod 600 /etc/NetworkManager/system-connections/*.nmconnection || true # Voorkom crash als map leeg is

# --- 1. Robotoo (Wi-Fi) instellingen ---
echo "Controleer Wi-Fi profiel: $WIFI_SSID..."
while nmcli connection show "$WIFI_SSID" > /dev/null 2>&1; do
    echo "bestaand '$WIFI_SSID' profiel gevonden, verwijderen..."
    nmcli connection delete "$WIFI_SSID"
done

echo "Nieuw profiel aanmaken"
nmcli connection add type wifi con-name "$WIFI_SSID" ifname "$WIFI_IFACE" ssid "$WIFI_SSID"

echo "Wi-Fi instellingen toepassen..."
nmcli connection modify "$WIFI_SSID" \
    ipv4.method manual \
    ipv4.addresses "172.18.108.12/24" \
    ipv4.gateway "172.18.108.1" \
    ipv4.dns "172.18.108.1,8.8.8.8" \
    connection.autoconnect yes \
    connection.autoconnect-priority 10 \
    connection.autoconnect-retries 0 \
    ipv4.may-fail no \
    wifi-sec.key-mgmt wpa-psk \
    wifi-sec.psk "$WIFI_PASS" \
    ipv4.route-metric 50

# --- 2. Power save (Globale instelling) uitschakelen ---
echo "Wi-Fi energiebesparing uitschakelen..."
WIFI_POWERSAVE_CONF="/etc/NetworkManager/conf.d/default-wifi-powersave-on.conf"
if [ -f "$WIFI_POWERSAVE_CONF" ]; then
    # Vervang 'wifi.powersave = 3' door 'wifi.powersave = 2'
    sed -i 's/wifi.powersave = [0-9]\+/wifi.powersave = 2/g' "$WIFI_POWERSAVE_CONF"
else
    echo "Waarschuwing: $WIFI_POWERSAVE_CONF niet gevonden. Handmatige configuratie mogelijk nodig."
fi

# --- 3. Connectivity check (20000 penalty) uitschakelen ---
echo "Connectivity check uitschakelen in NetworkManager.conf..."
NM_CONF="/etc/NetworkManager/NetworkManager.conf"
if ! grep -q "^\[connectivity\]" "$NM_CONF"; then
    echo -e "\n[connectivity]\nenabled=false" >> "$NM_CONF"
elif ! grep -q "^enabled=false" "$NM_CONF"; then
    # Als [connectivity] bestaat verwijder de regels en voeg enable = true toe
    sed -i '/^\[connectivity\]/,/^\[/{/^enabled=/d}' "$NM_CONF"
    sed -i '/^\[connectivity\]/a enabled=false' "$NM_CONF"
fi

# --- 4. Bestandsrechten herstellen ---
echo "Bestandsrechten van netwerkprofielen herstellen..."
chmod 600 /etc/NetworkManager/system-connections/*.nmconnection || true # Voorkom crash als map leeg is


# --- 5. Service herstarten en verbindingen activeren ---
echo "NetworkManager herstarten..."
systemctl restart NetworkManager

# Geef NetworkManager tijd om op te starten en interfaces te detecteren
sleep 10

echo "Verbindingen activeren..."
nmcli connection up "$WIFI_SSID" || echo "Let op: Kon Wi-Fi niet activeren. Controleer interface of bereik."

echo "=== Netwerkconfiguratie succesvol afgerond! ==="
