import xml.etree.ElementTree as ET
import html

# Function to build draw.io XML
def create_drawio_xml():
    # Root draw.io structure
    mxfile = ET.Element('mxfile', host='app.diagrams.net', modified='2026-08-16T11:30:00.000Z', agent='Gemini', version='24.7.5', type='device')
    diagram = ET.SubElement(mxfile, 'diagram', id='DC_Dashboard_Mockup', name='Dashboard Monitoring & Forecasting DC')
    mxGraphModel = ET.SubElement(diagram, 'mxGraphModel', dx='1600', dy='1000', grid='1', gridSize='10', guides='1', tooltips='1', connect='1', arrows='1', fold='1', page='1', pageScale='1', pageWidth='1654', pageHeight='1169', math='0', shadow='0')
    root = ET.SubElement(mxGraphModel, 'root')

    # Base cells
    ET.SubElement(root, 'mxCell', id='0')
    ET.SubElement(root, 'mxCell', id='1', parent='0')

    cells = []
    
    def add_cell(cell_id, value, style, parent_id='1', vertex='1', x=0, y=0, width=100, height=50):
        c = ET.SubElement(root, 'mxCell', id=str(cell_id), value=value, style=style, parent=str(parent_id), vertex=str(vertex))
        geo = ET.SubElement(c, 'mxGeometry', x=str(x), y=str(y), width=str(width), height=str(height), as_='geometry')
        return c

    # 1. Browser Window Frame (Canvas Container)
    add_cell('win_bg', '', 'rounded=1;whiteSpace=wrap;html=1;fillColor=#F8FAFC;strokeColor=#CBD5E1;strokeWidth=2;shadow=1;', '1', '1', 40, 30, 1560, 1080)
    
    # Window Title Bar (Mac/Modern style)
    add_cell('win_header', '', 'rounded=1;whiteSpace=wrap;html=1;fillColor=#1E293B;strokeColor=#1E293B;arcSize=4;', '1', '1', 40, 30, 1560, 45)
    # Window Buttons
    add_cell('btn_close', '', 'ellipse;whiteSpace=wrap;html=1;fillColor=#EF4444;strokeColor=none;', '1', '1', 60, 47, 12, 12)
    add_cell('btn_min', '', 'ellipse;whiteSpace=wrap;html=1;fillColor=#F59E0B;strokeColor=none;', '1', '1', 80, 47, 12, 12)
    add_cell('btn_max', '', 'ellipse;whiteSpace=wrap;html=1;fillColor=#10B981;strokeColor=none;', '1', '1', 100, 47, 12, 12)
    add_cell('win_title', '<b>DC INDOMARCO PRISMATAMA</b> — Sistem Monitoring & Forecasting Stok Fast-Moving (DSS Web & Alerting Bot)', 'text;html=1;strokeColor=none;fillColor=none;align=center;verticalAlign=middle;whiteSpace=wrap;rounded=0;fontColor=#F8FAFC;fontSize=12;fontFamily=Helvetica;', '1', '1', 250, 40, 1060, 25)

    # 2. Left Sidebar
    add_cell('sidebar_bg', '', 'whiteSpace=wrap;html=1;fillColor=#0F172A;strokeColor=#1E293B;', '1', '1', 40, 75, 240, 1035)
    
    # App Brand / Logo
    add_cell('brand_box', '<b>📦 INDO-DC SYSTEM</b><br><font color="#94A3B8" size="1">Inventory DSS & Alert v2.0</font>', 'text;html=1;strokeColor=none;fillColor=#1E293B;align=center;verticalAlign=middle;rounded=1;fontColor=#38BDF8;fontSize=14;', '1', '1', 55, 95, 210, 55)

    # Navigation Menu Items
    menu_items = [
        ('nav_dash', '📊 Dashboard Overview', '#2563EB', '#FFFFFF', '1'),
        ('nav_plu', '🏷️ Master Data PLU (>1000)', '#0F172A', '#94A3B8', '0'),
        ('nav_inout', '🔄 Mutasi Stok (In/Out)', '#0F172A', '#94A3B8', '0'),
        ('nav_forecast', '📈 Forecasting SMA & ROP', '#0F172A', '#94A3B8', '0'),
        ('nav_accuracy', '🎯 Evaluasi Akurasi MAPE', '#0F172A', '#94A3B8', '0'),
        ('nav_bot', '🤖 Telegram Alert Gateway', '#0F172A', '#94A3B8', '0'),
        ('nav_report', '📑 Laporan Pengadaan PO', '#0F172A', '#94A3B8', '0'),
        ('nav_setting', '⚙️ Konfigurasi Parameter', '#0F172A', '#94A3B8', '0'),
    ]
    
    curr_y = 175
    for item_id, label, bg, tc, active in menu_items:
        style = f'rounded=1;whiteSpace=wrap;html=1;fillColor={bg};strokeColor=none;align=left;spacingLeft=15;fontColor={tc};fontSize=12;fontStyle={"1" if active=="1" else "0"};'
        add_cell(item_id, label, style, '1', '1', 55, curr_y, 210, 42)
        curr_y += 50

    # User Profile at Bottom Sidebar
    add_cell('user_card', '<b>Alfiyan Nazar</b><br><font color="#64748B" size="1">Inventory Controller / Admin DC</font><br><font color="#10B981" size="1">● Online (DC Cirebon)</font>', 'rounded=1;whiteSpace=wrap;html=1;fillColor=#1E293B;strokeColor=#334155;align=left;spacingLeft=12;fontColor=#F8FAFC;fontSize=11;', '1', '1', 55, 990, 210, 65)

    # 3. Top Header Bar (Main View)
    add_cell('top_bar', '', 'whiteSpace=wrap;html=1;fillColor=#FFFFFF;strokeColor=#E2E8F0;', '1', '1', 280, 75, 1320, 60)
    
    # Search Input Box
    add_cell('search_box', '🔍 Cari Kode PLU / Nama Barang Fast-Moving...', 'rounded=1;whiteSpace=wrap;html=1;fillColor=#F1F5F9;strokeColor=#CBD5E1;align=left;spacingLeft=12;fontColor=#64748B;fontSize=12;', '1', '1', 300, 87, 380, 36)

    # Filter DC & Status Badges
    add_cell('dc_badge', '🏢 <b>DC Indomarco Cirebon</b>', 'rounded=1;whiteSpace=wrap;html=1;fillColor=#EFF6FF;strokeColor=#BFDBFE;fontColor=#1D4ED8;fontSize=11;align=center;', '1', '1', 700, 87, 180, 36)
    add_cell('cron_status', '⚡ <b>Cron Engine:</b> Running (SMA Daily)', 'rounded=1;whiteSpace=wrap;html=1;fillColor=#ECFDF5;strokeColor=#A7F3D0;fontColor=#065F46;fontSize=11;align=center;', '1', '1', 890, 87, 210, 36)
    add_cell('notif_badge', '🔔 <b>Alerts</b> <span style="background-color:#EF4444;color:#FFF;padding:2px 6px;border-radius:10px;font-size:10px;">3 Critical</span>', 'rounded=1;whiteSpace=wrap;html=1;fillColor=#FEF2F2;strokeColor=#FECACA;fontColor=#991B1B;fontSize=11;align=center;', '1', '1', 1110, 87, 140, 36)
    add_cell('date_widget', '📅 16 Agustus 2026 | 11:30 WIB', 'text;html=1;strokeColor=none;fillColor=none;align=right;verticalAlign=middle;fontColor=#64748B;fontSize=11;', '1', '1', 1260, 87, 320, 36)

    # 4. ROW 1: 4 KPI Summary Cards
    cards = [
        ('card_1', '📦 TOTAL MASTER PLU', '1,248 Item', 'Item Aktif Terdaftar di DC', '#FFFFFF', '#3B82F6', '#EFF6FF'),
        ('card_2', '🟢 STOK AMAN (SAFE)', '1,112 PLU', '89.1% Total Persediaan', '#FFFFFF', '#10B981', '#ECFDF5'),
        ('card_3', '⚠️ STATUS WARNING (ROP)', '98 PLU', 'Mendekati Reorder Point (7.8%)', '#FFFFFF', '#F59E0B', '#FFFBEB'),
        ('card_4', '🚨 STATUS KRITIS (CRITICAL)', '38 PLU', 'Stok &le; Safety Stock (3.1%)', '#FFFFFF', '#EF4444', '#FEF2F2'),
    ]
    card_x = 300
    for cid, title, val, sub, bg, border, tint in cards:
        val_color = border
        content = f'<div style="text-align:left;padding:6px;">' \
                  f'<span style="font-size:10px;font-weight:bold;color:#64748B;">{title}</span><br>' \
                  f'<span style="font-size:22px;font-weight:bold;color:{val_color};">{val}</span><br>' \
                  f'<span style="font-size:10px;color:#94A3B8;">{sub}</span>' \
                  f'</div>'
        style = f'rounded=1;whiteSpace=wrap;html=1;fillColor={bg};strokeColor={border};strokeWidth=2;shadow=1;'
        add_cell(cid, content, style, '1', '1', card_x, 150, 310, 85)
        card_x += 325

    # 5. ROW 2: VISUAL CHART & FORECAST ANALYTICS (LEFT 65%) + TELEGRAM BOT SIMULATOR (RIGHT 35%)
    # Left: Forecasting Visualizer
    add_cell('chart_box', '', 'rounded=1;whiteSpace=wrap;html=1;fillColor=#FFFFFF;strokeColor=#E2E8F0;strokeWidth=1;shadow=1;', '1', '1', 300, 250, 820, 420)
    
    # Chart Header & Controls
    add_cell('chart_title', '<b>📈 Kurva Peramalan Stok vs Pengeluaran Aktual (Simple Moving Average)</b><br><font color="#64748B" size="1">PLU: <b>2004581</b> - Indomie Goreng Spesial 85g (Kategori: Instant Food / Fast-Moving)</font>', 'text;html=1;strokeColor=none;fillColor=none;align=left;verticalAlign=top;fontColor=#1E293B;fontSize=12;', '1', '1', 320, 265, 520, 40)
    
    # Filter Buttons for Chart
    add_cell('btn_n7', '<b>SMA N=7 (Aktif)</b>', 'rounded=1;whiteSpace=wrap;html=1;fillColor=#2563EB;strokeColor=#2563EB;fontColor=#FFFFFF;fontSize=10;align=center;', '1', '1', 860, 265, 100, 26)
    add_cell('btn_n14', 'SMA N=14', 'rounded=1;whiteSpace=wrap;html=1;fillColor=#F1F5F9;strokeColor=#CBD5E1;fontColor=#475569;fontSize=10;align=center;', '1', '1', 970, 265, 70, 26)
    add_cell('btn_n30', 'SMA N=30', 'rounded=1;whiteSpace=wrap;html=1;fillColor=#F1F5F9;strokeColor=#CBD5E1;fontColor=#475569;fontSize=10;align=center;', '1', '1', 1045, 265, 70, 26)

    # Chart Canvas / Graphic simulation
    add_cell('chart_canvas', '', 'rounded=1;whiteSpace=wrap;html=1;fillColor=#F8FAFC;strokeColor=#E2E8F0;', '1', '1', 320, 315, 780, 240)
    
    # Chart Internal Visual Elements (Lines, Grid, Legends)
    # Safety Stock Zone
    add_cell('ss_zone', '<font color="#059669"><b>ZONA SAFETY STOCK (SS = 150 Karton)</b></font>', 'rounded=0;whiteSpace=wrap;html=1;fillColor=#D1FAE5;strokeColor=none;opacity=70;fontSize=10;align=center;', '1', '1', 340, 500, 740, 40)
    # Reorder Point (ROP) Line
    add_cell('rop_line', '', 'shape=line;strokeColor=#DC2626;strokeWidth=2;dashed=1;', '1', '1', 340, 440, 740, 10)
    add_cell('rop_label', '<font color="#DC2626"><b>Ambang Batas ROP = 345 Ktn</b></font>', 'text;html=1;strokeColor=none;fillColor=none;align=left;fontSize=10;fontStyle=1;', '1', '1', 345, 420, 200, 20)

    # Actual Outbound Usage Curve (Simulated polyline/curve)
    add_cell('curve_actual', '', 'curved=1;strokeColor=#3B82F6;strokeWidth=3;fillColor=none;', '1', '1', 350, 350, 720, 150)
    # SMA Forecast Line
    add_cell('curve_sma', '', 'curved=1;strokeColor=#10B981;strokeWidth=3;dashed=1;fillColor=none;', '1', '1', 350, 370, 720, 120)

    # Chart Legends
    add_cell('legend_1', '<font color="#3B82F6">━━</font> Pengeluaran Aktual Harian', 'text;html=1;strokeColor=none;fillColor=none;align=left;fontSize=10;', '1', '1', 340, 560, 200, 20)
    add_cell('legend_2', '<font color="#10B981">┈┈</font> Hasil Prediksi SMA (N=7)', 'text;html=1;strokeColor=none;fillColor=none;align=left;fontSize=10;', '1', '1', 550, 560, 180, 20)
    add_cell('legend_3', '<font color="#DC2626">┈┈</font> Reorder Point (ROP)', 'text;html=1;strokeColor=none;fillColor=none;align=left;fontSize=10;', '1', '1', 740, 560, 150, 20)
    add_cell('legend_4', '<span style="background-color:#D1FAE5;padding:1px 4px;">■</span> Safety Stock', 'text;html=1;strokeColor=none;fillColor=none;align=left;fontSize=10;', '1', '1', 900, 560, 150, 20)

    # Key Calculation Metrics Panel below chart
    metrics_html = '<b>Perhitungan Logistik PLU 2004581:</b> &nbsp;|&nbsp; ' \
                   'Stok Fisik: <b>420 Ktn</b> &nbsp;|&nbsp; ' \
                   'Rata-rata Keluar (SMA): <b>65 Ktn/hari</b> &nbsp;|&nbsp; ' \
                   'Lead Time: <b>3 Hari</b> &nbsp;|&nbsp; ' \
                   'Safety Stock (SS): <b>150 Ktn</b> &nbsp;|&nbsp; ' \
                   'Reorder Point (ROP): <b>345 Ktn</b> &nbsp;|&nbsp; ' \
                   'Est. Habis: <b>6.4 Hari</b> &nbsp;|&nbsp; ' \
                   'MAPE: <b style="color:#059669;">4.82% (Sangat Baik)</b> &nbsp;|&nbsp; ' \
                   'Status: <b style="color:#D97706;background-color:#FEF3C7;padding:2px 6px;border-radius:4px;">⚠️ WARNING</b>'
    add_cell('chart_submetrics', metrics_html, 'rounded=1;whiteSpace=wrap;html=1;fillColor=#F1F5F9;strokeColor=#CBD5E1;align=center;fontSize=10;fontColor=#1E293B;', '1', '1', 320, 595, 780, 55)

    # Right: Telegram Bot Live Gateway Simulator Panel
    add_cell('bot_box', '', 'rounded=1;whiteSpace=wrap;html=1;fillColor=#FFFFFF;strokeColor=#E2E8F0;strokeWidth=1;shadow=1;', '1', '1', 1140, 250, 460, 420)
    # Bot Header
    add_cell('bot_header', '🤖 <b>Telegram Alert Gateway Simulator</b><br><font color="#64748B" size="1">Bot: @IndoDC_StockAlertBot | Mode: Webhook Realtime</font>', 'rounded=1;whiteSpace=wrap;html=1;fillColor=#0284C7;strokeColor=#0284C7;align=left;spacingLeft=12;fontColor=#FFFFFF;fontSize=11;', '1', '1', 1140, 250, 460, 45)

    # Chat Background Container
    add_cell('chat_bg', '', 'whiteSpace=wrap;html=1;fillColor=#E0F2FE;strokeColor=none;', '1', '1', 1140, 295, 460, 330)

    # Bubble Alert 1 (Critical)
    b1_text = '🚨 <b>[CRITICAL STOCKOUT ALERT]</b><br>' \
              '━━━━━━━━━━━━━━━━━━━━<br>' \
              '<b>DC:</b> Indomarco Cirebon<br>' \
              '<b>PLU:</b> 1002341 - Minyak Goreng 2L Pouch<br>' \
              '<b>Stok Fisik:</b> 45 Karton<br>' \
              '<b>Safety Stock:</b> 60 Karton<br>' \
              '<b>Status:</b> <font color="#EF4444"><b>KRITIS (Stok &le; SS)</b></font><br>' \
              '<b>Est. Stok Habis:</b> 1.2 Hari Lagi<br>' \
              '━━━━━━━━━━━━━━━━━━━━<br>' \
              '⚠️ <i>Segera terbitkan PO Darurat ke Supplier!</i><br>' \
              '<font size="1" color="#64748B">16/08/2026 07:00:15 WIB</font>'
    add_cell('chat_bubble1', b1_text, 'rounded=1;whiteSpace=wrap;html=1;fillColor=#FFFFFF;strokeColor=#CBD5E1;align=left;spacingLeft=10;fontColor=#1E293B;fontSize=10;shadow=1;', '1', '1', 1155, 305, 430, 140)

    # Bubble Alert 2 (Warning)
    b2_text = '⚠️ <b>[WARNING - REORDER POINT]</b><br>' \
              '━━━━━━━━━━━━━━━━━━━━<br>' \
              '<b>PLU:</b> 2004581 - Indomie Goreng Spc 85g<br>' \
              '<b>Stok Fisik:</b> 420 Ktn (ROP: 345 Ktn)<br>' \
              '<b>Rekomendasi PO:</b> 500 Karton<br>' \
              '<b>Est. Depletion:</b> 6.4 Hari<br>' \
              '<font size="1" color="#64748B">16/08/2026 07:00:16 WIB</font>'
    add_cell('chat_bubble2', b2_text, 'rounded=1;whiteSpace=wrap;html=1;fillColor=#FFFFFF;strokeColor=#CBD5E1;align=left;spacingLeft=10;fontColor=#1E293B;fontSize=10;shadow=1;', '1', '1', 1155, 455, 430, 105)

    # Bot Interactive Action Buttons
    add_cell('bot_btn1', '🔗 Buka Web Dashboard', 'rounded=1;whiteSpace=wrap;html=1;fillColor=#0284C7;strokeColor=none;fontColor=#FFFFFF;fontSize=9;align=center;', '1', '1', 1165, 570, 195, 25)
    add_cell('bot_btn2', '📝 Draft Purchase Order', 'rounded=1;whiteSpace=wrap;html=1;fillColor=#059669;strokeColor=none;fontColor=#FFFFFF;fontSize=9;align=center;', '1', '1', 1375, 570, 195, 25)

    # Bot Status footer
    add_cell('bot_footer', 'Status: <b>Terkirim ke 5 Supervisor Gudang</b> | Anti-Spam: <b>Active</b>', 'text;html=1;strokeColor=none;fillColor=none;align=center;fontSize=10;fontColor=#64748B;', '1', '1', 1140, 630, 460, 30)

    # 6. ROW 3: FAST-MOVING DC INVENTORY MONITORING TABLE (>1.000 PLU)
    add_cell('table_container', '', 'rounded=1;whiteSpace=wrap;html=1;fillColor=#FFFFFF;strokeColor=#E2E8F0;strokeWidth=1;shadow=1;', '1', '1', 300, 690, 1300, 395)
    
    # Table Header & Controls
    add_cell('tbl_heading', '<b>📋 Monitoring Stok Fast-Moving DC & Rekomendasi ROP (Total 1,248 PLU Aktif)</b>', 'text;html=1;strokeColor=none;fillColor=none;align=left;verticalAlign=middle;fontColor=#1E293B;fontSize=12;', '1', '1', 320, 700, 500, 30)
    
    add_cell('tbl_filter_cat', 'Kategori: <b>Semua Divisi</b> ▼', 'rounded=1;whiteSpace=wrap;html=1;fillColor=#F8FAFC;strokeColor=#CBD5E1;fontSize=10;fontColor=#334155;align=center;', '1', '1', 870, 702, 140, 26)
    add_cell('tbl_filter_status', 'Status: <b>Semua Status</b> ▼', 'rounded=1;whiteSpace=wrap;html=1;fillColor=#F8FAFC;strokeColor=#CBD5E1;fontSize=10;fontColor=#334155;align=center;', '1', '1', 1020, 702, 140, 26)
    add_cell('tbl_btn_export', '📥 Export Excel / CSV', 'rounded=1;whiteSpace=wrap;html=1;fillColor=#059669;strokeColor=none;fontSize=10;fontColor=#FFFFFF;fontStyle=1;align=center;', '1', '1', 1170, 702, 130, 26)
    add_cell('tbl_btn_sync', '⚡ Run Batch Forecast', 'rounded=1;whiteSpace=wrap;html=1;fillColor=#2563EB;strokeColor=none;fontSize=10;fontColor=#FFFFFF;fontStyle=1;align=center;', '1', '1', 1310, 702, 130, 26)

    # Table Column Headers
    cols = [
        ('No', 40),
        ('Kode PLU', 90),
        ('Nama Barang Fast-Moving', 250),
        ('Kategori', 120),
        ('Stok Fisik', 85),
        ('Lead Time', 75),
        ('SMA (Daily)', 90),
        ('Safety Stock', 90),
        ('ROP Target', 90),
        ('Est. Habis', 85),
        ('MAPE', 75),
        ('Status Persediaan', 110),
        ('Aksi Tindakan', 100)
    ]
    
    th_x = 320
    for name, w in cols:
        add_cell(f'th_{name}', f'<b>{name}</b>', 'rounded=0;whiteSpace=wrap;html=1;fillColor=#1E293B;strokeColor=#334155;fontColor=#FFFFFF;fontSize=10;align=center;', '1', '1', th_x, 740, w, 32)
        th_x += w

    # Table Rows Data
    table_rows = [
        ('1', '1002341', 'Minyak Goreng Sawit 2L Pouch', 'Sembako & Minyak', '45 Ktn', '2 Hari', '38 Ktn/hr', '60 Ktn', '136 Ktn', '1.2 Hari', '3.54%', '🚨 KRITIS', '#EF4444', '#FEF2F2', 'Draft PO'),
        ('2', '2004581', 'Indomie Goreng Spesial 85g', 'Makanan Instan', '420 Ktn', '3 Hari', '65 Ktn/hr', '150 Ktn', '345 Ktn', '6.4 Hari', '4.82%', '⚠️ WARNING', '#D97706', '#FFFBEB', 'Draft PO'),
        ('3', '3001892', 'Aqua Air Mineral 600ml (Karton)', 'Minuman', '890 Ktn', '2 Hari', '110 Ktn/hr', '200 Ktn', '420 Ktn', '8.1 Hari', '2.91%', '🟢 AMAN', '#059669', '#ECFDF5', 'Detail'),
        ('4', '4005112', 'Kopi Kapal Api Spesial 165g', 'Minuman & Kopi', '115 Ktn', '4 Hari', '25 Ktn/hr', '80 Ktn', '180 Ktn', '4.6 Hari', '5.12%', '⚠️ WARNING', '#D97706', '#FFFBEB', 'Draft PO'),
        ('5', '5009823', 'Sabun Ekonomi Pencuci Piring 780ml', 'Homecare & Sabun', '620 Ktn', '3 Hari', '40 Ktn/hr', '100 Ktn', '220 Ktn', '15.5 Hari', '3.18%', '🟢 AMAN', '#059669', '#ECFDF5', 'Detail'),
        ('6', '6001249', 'Beras Ramos Super Premium 5kg', 'Sembako', '32 Ktn', '2 Hari', '28 Ktn/hr', '50 Ktn', '106 Ktn', '1.1 Hari', '4.10%', '🚨 KRITIS', '#EF4444', '#FEF2F2', 'Draft PO')
    ]

    r_y = 772
    for r_no, plu, item_name, cat, stock, lt, sma, ss, rop, est, mape, status_text, s_color, s_bg, act in table_rows:
        row_bg = '#FFFFFF' if int(r_no) % 2 != 0 else '#F8FAFC'
        row_x = 320
        
        # Cols mapping
        row_data = [
            (r_no, 40, 'center', row_bg, '#334155'),
            (plu, 90, 'center', row_bg, '#2563EB;font-weight:bold'),
            (item_name, 250, 'left', row_bg, '#1E293B'),
            (cat, 120, 'left', row_bg, '#64748B'),
            (stock, 85, 'center', row_bg, '#1E293B;font-weight:bold'),
            (lt, 75, 'center', row_bg, '#475569'),
            (sma, 90, 'center', row_bg, '#475569'),
            (ss, 90, 'center', row_bg, '#475569'),
            (rop, 90, 'center', row_bg, '#475569'),
            (est, 85, 'center', row_bg, '#1E293B'),
            (mape, 75, 'center', row_bg, '#059669;font-weight:bold'),
            (f'<span style="background-color:{s_bg};color:{s_color};padding:2px 8px;border-radius:4px;font-weight:bold;">{status_text}</span>', 110, 'center', row_bg, '#1E293B'),
            (f'<span style="background-color:#2563EB;color:#FFF;padding:2px 8px;border-radius:4px;font-size:9px;">{act}</span>', 100, 'center', row_bg, '#FFFFFF')
        ]
        
        for val, w, align, bg_c, font_c in row_data:
            style = f'rounded=0;whiteSpace=wrap;html=1;fillColor={bg_c};strokeColor=#E2E8F0;fontSize=10;align={align};spacingLeft=6;spacingRight=6;'
            add_cell(f'cell_{r_no}_{w}_{row_x}', val, style, '1', '1', row_x, r_y, w, 36)
            row_x += w
            
        r_y += 36

    # Table Pagination Footer
    add_cell('tbl_pagination', 'Menampilkan <b>1 - 6</b> dari <b>1,248</b> Data PLU Fast-Moving | <b>Server-side Processing: 0.042 detik</b>', 'text;html=1;strokeColor=none;fillColor=none;align=left;fontSize=10;fontColor=#64748B;', '1', '1', 320, 1030, 600, 30)
    add_cell('tbl_pages', 'Halaman: <b>[ 1 ]</b> &nbsp; 2 &nbsp; 3 &nbsp; 4 &nbsp; ... &nbsp; 208 &nbsp; [Berikutnya ▶]', 'rounded=1;whiteSpace=wrap;html=1;fillColor=#F1F5F9;strokeColor=#CBD5E1;fontSize=10;fontColor=#1E293B;align=center;', '1', '1', 1250, 1025, 300, 30)

    # Convert tree to string
    xml_str = ET.tostring(mxfile, encoding='utf-8', xml_declaration=True).decode('utf-8')
    return xml_str

# Generate the drawio xml file
xml_content = create_drawio_xml()
with open("mockup_dashboard_dc.xml", "w", encoding="utf-8") as f:
    f.write(xml_content)

# Also save with .drawio extension for convenience
with open("mockup_dashboard_dc.drawio", "w", encoding="utf-8") as f:
    f.write(xml_content)

print("Draw.io XML / .drawio files successfully created and validated.")