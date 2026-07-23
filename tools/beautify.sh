#!/bin/bash
# Make XFCE look like macOS (WhiteSur) — pretty but still light. Runs for user 'guest'.
set -e
export DEBIAN_FRONTEND=noninteractive

echo "[1/5] Cai goi ho tro..."
apt-get update -y
apt-get install -y git plank papirus-icon-theme sassc gtk2-engines-murrine \
    gnome-themes-extra fonts-noto libglib2.0-dev-bin

cd /root
echo "[2/5] Tai WhiteSur GTK theme..."
rm -rf WhiteSur-gtk-theme WhiteSur-icon-theme
git clone --depth=1 https://github.com/vinceliuice/WhiteSur-gtk-theme.git
git clone --depth=1 https://github.com/vinceliuice/WhiteSur-icon-theme.git

echo "[3/5] Cai theme + icon (toan he thong)..."
./WhiteSur-gtk-theme/install.sh -c Dark -t default >/dev/null 2>&1 || ./WhiteSur-gtk-theme/install.sh >/dev/null 2>&1
./WhiteSur-icon-theme/install.sh >/dev/null 2>&1

# Tim ten theme/icon that su da cai
GTK=$(ls -d /usr/share/themes/WhiteSur-Dark* 2>/dev/null | head -1 | xargs -n1 basename)
[ -z "$GTK" ] && GTK=$(ls -d /usr/share/themes/WhiteSur* 2>/dev/null | head -1 | xargs -n1 basename)
ICON=$(ls -d /usr/share/icons/WhiteSur-dark* 2>/dev/null | head -1 | xargs -n1 basename)
[ -z "$ICON" ] && ICON="Papirus-Dark"
echo "    GTK=$GTK  ICON=$ICON"

echo "[4/5] Ap config cho user guest..."
GU=/home/guest
mkdir -p $GU/.config/xfce4/xfconf/xfce-perchannel-xml $GU/.config/autostart

cat > $GU/.config/xfce4/xfconf/xfce-perchannel-xml/xsettings.xml <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<channel name="xsettings" version="1.0">
  <property name="Net" type="empty">
    <property name="ThemeName" type="string" value="$GTK"/>
    <property name="IconThemeName" type="string" value="$ICON"/>
  </property>
  <property name="Gtk" type="empty">
    <property name="FontName" type="string" value="Noto Sans 10"/>
    <property name="CursorThemeName" type="string" value="Adwaita"/>
  </property>
</channel>
EOF

cat > $GU/.config/xfce4/xfconf/xfce-perchannel-xml/xfwm4.xml <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<channel name="xfwm4" version="1.0">
  <property name="general" type="empty">
    <property name="theme" type="string" value="$GTK"/>
    <property name="title_font" type="string" value="Noto Sans Bold 10"/>
    <property name="button_layout" type="string" value="O|SHMC"/>
  </property>
</channel>
EOF

cat > $GU/.config/autostart/plank.desktop <<EOF
[Desktop Entry]
Type=Application
Name=Plank
Exec=plank
X-GNOME-Autostart-enabled=true
EOF

# Wallpaper WhiteSur (neu co)
WP=$(ls /usr/share/backgrounds/*WhiteSur* 2>/dev/null | head -1)
[ -z "$WP" ] && WP=$(ls /root/WhiteSur-gtk-theme/wallpaper/*.jpg 2>/dev/null | head -1)
if [ -n "$WP" ]; then cp "$WP" $GU/wallpaper.jpg 2>/dev/null || true; fi

chown -R guest:guest $GU/.config $GU/wallpaper.jpg 2>/dev/null || true

echo "[5/5] Restart xrdp..."
systemctl restart xrdp
echo "BEAUTIFY_DONE GTK=$GTK ICON=$ICON"
