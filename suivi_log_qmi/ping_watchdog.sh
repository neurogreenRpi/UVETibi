#!/bin/bash
# ============================================================
# ping_watchdog.sh - Surveillance de connectivité réseau
# Ping Google toutes les X secondes, log le résultat,
# et redémarre la Pi en cas d'échec.
# ============================================================

# --- Configuration ---
TARGET="8.8.8.8"           # Cible du ping (DNS Google)
INTERVAL=60                 # Intervalle entre chaque ping (secondes)
PING_COUNT=3                # Nombre de paquets ping envoyés
PING_TIMEOUT=10             # Timeout par ping (secondes)
LOG_FILE="/var/log/ping_watchdog.log"
MAX_LOG_LINES=10000         # Rotation : garder les N dernières lignes

# --- Fonction de logging ---
log() {
    local STATUS="$1"
    local MESSAGE="$2"
    local TIMESTAMP
    TIMESTAMP=$(date "+%Y-%m-%d %H:%M:%S")
    echo "${TIMESTAMP} | ${STATUS} | ${MESSAGE}" | tee -a "$LOG_FILE"
}

# --- Rotation du log ---
rotate_log() {
    if [ -f "$LOG_FILE" ]; then
        local LINES
        LINES=$(wc -l < "$LOG_FILE")
        if [ "$LINES" -gt "$MAX_LOG_LINES" ]; then
            # Garde uniquement les dernières MAX_LOG_LINES lignes
            tail -n "$MAX_LOG_LINES" "$LOG_FILE" > "${LOG_FILE}.tmp"
            mv "${LOG_FILE}.tmp" "$LOG_FILE"
            log "INFO" "Rotation du log effectuée (>${MAX_LOG_LINES} lignes)"
        fi
    fi
}

# --- Vérification des droits root (nécessaire pour reboot) ---
if [ "$EUID" -ne 0 ]; then
    echo "ATTENTION : Ce script doit être lancé en root pour pouvoir redémarrer la Pi."
    echo "Relancez avec : sudo $0"
    exit 1
fi

# --- Initialisation ---
log "INFO" "=== Démarrage du ping watchdog | Cible: ${TARGET} | Intervalle: ${INTERVAL}s ==="

# --- Boucle principale ---
while true; do

    # Effectue le ping
    if ping -c "$PING_COUNT" -W "$PING_TIMEOUT" "$TARGET" > /dev/null 2>&1; then
        log "OK" "Ping vers ${TARGET} réussi"
    else
        log "ERREUR" "Ping vers ${TARGET} ÉCHOUÉ — Redémarrage de la Pi dans 5 secondes..."
        sleep 5
        log "ERREUR" "Exécution du reboot..."
        /sbin/reboot
    fi

    # Rotation du log si nécessaire
    rotate_log

    # Attente avant le prochain ping
    sleep "$INTERVAL"

done
