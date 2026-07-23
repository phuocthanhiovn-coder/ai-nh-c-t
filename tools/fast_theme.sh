#!/bin/bash
# Fast reliable pretty theme: Arc-Dark + Papirus + Plank dock. No SCSS build.
export DEBIAN_FRONTEND=noninteractive
pkill -f beautify.sh 2>/dev/null
pkill sassc 2>/dev/null
apt-get install -y arc-theme plank papirus-icon-theme >/dev/null 2>&1

GTK=Arc-Dark
ICON=Papirus-Dark
GU=/home/guest
mkdir -p "$GU/.config/xfce4/xfconf/xfce-perchannel-xml" "$GU/.config/autostart"

cat > "$GU/.config/xfce4/xfconf/xfce-perchannel-xml/xsettings.xml" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<channel name="xsettings" version="1.0">
  <property name="Net" type="empty">
    <property name="ThemeName" type="string" value="$GTK"/>
    <property name="IconThemeName" type="string" value="$ICON"/>
  </property>
  <property name="Gtk" type="empty">
    <property name="FontName" type="string" value="Noto Sans 10"/>
  </property>
</channel>
EOF

cat > "$GU/.config/xfce4/xfconf/xfce-perchannel-xml/xfwm4.xml" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<channel name="xfwm4" version="1.0">
  <property name="general" type="empty">
    <property name="theme" type="string" value="$GTK"/>
    <property name="title_font" type="string" value="Noto Sans Bold 10"/>
  </property>
</channel>
EOF

cat > "$GU/.config/autostart/plank.desktop" <<EOF
[Desktop Entry]
Type=Application
Name=Plank
Exec=plank
X-GNOME-Autostart-enabled=true
EOF

chown -R guest:guest "$GU/.config"
systemctl restart xrdp
echo "FAST_THEME_DONE GTK=$GTK ICON=$ICON"
ls /usr/share/themes/ | grep -i arc
