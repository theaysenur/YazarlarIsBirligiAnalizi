import sys
import threading
import webbrowser
import ast
import json
import os

import pandas as pd
import networkx as nx
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State

import dash_cytoscape as cyto
import math

# --------------------------------------------------------
# stdout'un (terminal) UTF-8 formatında yazması için ek kontrol.
# Bazı ortamlarda stdout encoding'i UTF-8 olmayabiliyor.
# sys.stdout.encoding.lower() 'utf-8' değilse, yeniden tanımlıyoruz.
# --------------------------------------------------------
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

print("Başlangıç: Program başladı.")

# -----------------------------------------------------
# Node ve GraphData Sınıfları
# -----------------------------------------------------
class Node:
    """
    Bu sınıf, her bir 'yazar' (düğüm) bilgisini tutar.
    name: yazarın temizlenmiş adı (örneğin 'ali veli' gibi).
    edges: yazarın komşuları (ortak yazarlar) ve aralarındaki 'weight' değeri.
    papers: yazarın yer aldığı makale bilgilerini (title, doi, vb.) tutar.
    all_authors: yazarın işbirliği yaptığı tüm yazarları tutan bir set.
    author_position: 1 => baş yazar (ORCID dolu), 0 => normal yazar.
    """
    def __init__(self, name):
        """
        name: yazarın "temizlenmiş" adı (orcid vb. dikkate alınmadan).
        """
        self.name = name
        self.edges = {}  # Örn: {'mehmet': {'weight':2}, 'ayşe': {'weight':1}} gibi.
        self.papers = []
        self.all_authors = set()
        self.author_position = 0  # Varsayılan '0', eğer ORCID dolu ise '1' olacak.

    def add_edge(self, neighbor_name):
        """
        Bir yazar ile arasına kenar ekler veya mevcutsa 'weight'i arttırır.
        neighbor_name: komşu yazarın adı.
        """
        key = neighbor_name.lower()
        if key not in self.edges:
            # Eğer edge(kenar) yoksa 1 olarak başlat
            self.edges[key] = {'weight': 1}
        else:
            # Varsa, her tekrar işbirliği bulunduğunda artır.
            self.edges[key]['weight'] += 1

    def add_paper(self, paper_title, doi, author_position, main_author=None, authors=None):
        """
        Yazarın makalelerine yeni bir makale bilgisi ekler.
        Aynı DOI varsa tekrar eklemeyi engeller.
        author_position: 1 => baş yazar, 0 => normal yazar.
        main_author: makaledeki ana yazarın ismi
        authors: makaledeki tüm yazarların listesi
        """
        # Aynı DOI'ye sahip makale varsa eklememek için kontrol
        if not any(paper[1] == doi for paper in self.papers):
            self.papers.append((paper_title, doi, author_position, main_author, authors))
            if authors:
                # Makale yazarlarının hepsini 'all_authors' setine ekliyoruz
                self.all_authors.update([author.lower() for author in authors if author])

        # Eğer sonradan bu düğüm baş yazar (author_position=1) olarak işaretlenirse,
        # en güncel konumu '1' olarak tutalım.
        if author_position == 1:
            self.author_position = 1


class GraphData:
    """
    Bu sınıf, tüm yazar düğümlerini ('Node' nesneleri) ve 
    ID haritalama dosyasını (JSON) yönetir.
    """
    def __init__(self, id_mapping_file='author_id_map.json'):
        # nodes sözlüğü: {'ali veli': Node('ali veli'), 'mehmet': Node('mehmet')} gibi
        self.nodes = {}
        self.id_mapping_file = id_mapping_file
        self.load_id_mapping()

    def load_id_mapping(self):
        """
        JSON dosyasından (author_id_map.json) kaydedilmiş yazar isimlerini yükler.
        """
        if os.path.exists(self.id_mapping_file):
            try:
                with open(self.id_mapping_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # JSON içinde saklı "nodes" anahtarındaki yazarların her biri için Node oluştur.
                    self.nodes = {k.lower(): Node(k) for k in data.get('nodes', [])}
                print("ID haritası yüklendi.")
            except json.JSONDecodeError:
                print("Hata: 'author_id_map.json' dosyası geçersiz JSON formatında. Dosya sıfırlanıyor.")
                self.nodes = {}
                self.save_id_mapping()
            except Exception as e:
                print(f"Hata: 'author_id_map.json' dosyası okunurken bir hata oluştu: {e}")
                self.nodes = {}
                self.save_id_mapping()
        else:
            print("ID haritası dosyası bulunamadı. Yeni ID haritası oluşturuluyor.")
            self.nodes = {}
            self.save_id_mapping()

    def save_id_mapping(self):
        """
        Mevcut 'nodes' sözlüğündeki yazarları, JSON dosyasına kaydeder.
        Dosyada sadece yazar isimleri tutulur.
        """
        data = {
            'nodes': [node.name for node in self.nodes.values()]
        }
        with open(self.id_mapping_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        print("ID haritası kaydedildi.")

    def add_node(self, name, author_position=0):
        """
        Verilen isimde bir Node (yazar) yoksa ekler, varsa 
        gerekirse author_position'ı günceller.
        """
        key = name.lower()
        if key not in self.nodes:
            node = Node(name)
            node.author_position = author_position
            self.nodes[key] = node
            print(f"Yeni yazar eklendi: {name}")
        else:
            # Daha önce eklenmiş yazar sonradan 'baş yazar' olacaksa 1 yap
            if author_position == 1:
                self.nodes[key].author_position = 1
        return self.nodes[key]

    def add_author(self, name, author_position=0):
        """
        add_node'u çağırarak yazar ekleme işini yapar (küçük bir sarmala).
        """
        return self.add_node(name, author_position=author_position)

    def add_paper(self, paper_title, doi, main_author_name, author_position, coauthors,
                  all_authors_info, authors=None):
        """
        Bir makale bilgisini grafiğe ekler.
          - Baş yazar eklenir
          - Co-author'lar eklenir
          - Kenarlar (baş yazar <-> coauthor) eklenir
        """
        # Baş yazar ekleniyor
        main_author = self.add_author(main_author_name, author_position=author_position)
        main_author.add_paper(paper_title, doi, author_position, main_author.name, authors=authors)

        # Coauthor'lar işleniyor
        if pd.notna(coauthors):
            try:
                coauthor_names = ast.literal_eval(coauthors)
                if not isinstance(coauthor_names, list):
                    coauthor_names = [coauthors]
            except (ValueError, SyntaxError):
                # coauthors hücresinde list formatı yoksa ',' ile ayırmayı deniyoruz
                coauthor_names = [name.strip() for name in coauthors.split(',') if name.strip()]
        else:
            coauthor_names = []

        # Hepsinin metin olarak temizlenmesi (lower, strip)    
        coauthor_names = [clean_text(name) for name in coauthor_names if name]

        # Makaledeki tüm yazarlar: baş yazar + coauthor'lar
        all_authors = [main_author_name] + coauthor_names

        # Coauthor'ları ekle
        for coauthor_name in coauthor_names:
            self.add_author(coauthor_name, author_position=0)

        # Baş yazarla her coauthor arasında edge oluştur
        for coauthor_name in coauthor_names:
            main_author.add_edge(coauthor_name)
            self.nodes[coauthor_name.lower()].add_edge(main_author.name)

        # Makaleyi coauthor'lara ekle
        for coauthor_name in coauthor_names:
            coauthor = self.nodes[coauthor_name.lower()]
            coauthor.add_paper(paper_title, doi, 0, main_author.name, authors=authors)


# -----------------------------------------------------
# Metin Temizleme Fonksiyonu
# -----------------------------------------------------
def clean_text(text):
    """
    Bir metnin baştaki/sondaki boşluklarını siler
    ve küçük harfe dönüştürür.
    """
    if isinstance(text, str):
        return text.strip().lower()
    return str(text).strip().lower()


# -----------------------------------------------------
# Excel'den Veri Yükleme Fonksiyonu (ORCID Entegrasyonu)
# -----------------------------------------------------
def load_data(file_path):
    """
    Parametre:
      file_path: Excel dosyasının tam yolu.
    Dönüş:
      Bir pandas DataFrame veya None (hata durumunda).
    """
    try:
        # Excel'in ilk sayfasını okuyoruz (sheet_name=0).
        df_full = pd.read_excel(file_path, sheet_name=0)
        # İsteğe göre, çok büyük dosyalarda test yapmak için ilk 1000 satırı alıyoruz.
        df = df_full.head(1000).copy()

        # Sütun adlarını trimleyip küçük harfe dönüştürüyoruz.
        df.columns = [col.strip().lower() for col in df.columns]
        print("Excel dosyasındaki sütunlar:")
        print(df.columns.tolist())

        # Gerekli sütunların varlığını kontrol ediyoruz.
        required_columns = ['doi', 'author_name', 'coauthors', 'paper_title', 'orcid']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Excel dosyasında eksik sütunlar var: {', '.join(missing_columns)}")

        print("Veri başarıyla yüklendi.")
        print(f"Okunan satır sayısı: {df.shape[0]}")
        print("Verinin ilk 5 satırı:")
        print(df.head())

        # ORCID sütunu boş olanları boş string yapalım
        df['orcid'] = df['orcid'].fillna('')

        # ORCID'i dolu olanları => author_position=1, boş olanları => 0
        # Yani ORCID'i varsa o yazar "baş yazar" olarak işaretleniyor.
        df['author_position'] = df.apply(
            lambda row: 1 if row['orcid'].strip() != '' else 0,
            axis=1
        )

        print("Güncellenmiş veri (ORCID -> author_position):")
        print(df[['author_name','orcid','author_position']].head())

        return df
    except FileNotFoundError:
        print(f"Hata: Excel dosyası bulunamadı. Lütfen dosya yolunu kontrol edin: {file_path}")
        return None
    except ValueError as ve:
        print(f"Hata: {ve}")
        return None
    except Exception as e:
        print(f"Beklenmeyen bir hata oluştu: {e}")
        return None


# -----------------------------------------------------
# Grafik Oluşturma Fonksiyonu
# -----------------------------------------------------
def create_graph(df):
    """
    Verilen DataFrame içindeki satırları okuyup
    Node'lar ve edge'ler (işbirliği ilişkisi) oluşturarak
    bir GraphData nesnesi döndürür.
    """
    graph = GraphData()
    all_authors_info = {}

    for idx, row in df.iterrows():
        paper_title = row['paper_title']
        doi = row['doi']
        main_author_name = clean_text(row['author_name'])
        author_position = row['author_position']  # 1 => baş yazar, 0 => normal
        coauthors = row['coauthors']

        # coauthors sütunu list gibi parse edelim
        if pd.notna(coauthors):
            try:
                coauthor_names = ast.literal_eval(coauthors)
                if not isinstance(coauthor_names, list):
                    coauthor_names = [coauthors]
            except (ValueError, SyntaxError):
                coauthor_names = [name.strip() for name in coauthors.split(',') if name.strip()]
            coauthor_names = [clean_text(name) for name in coauthor_names if name]
            authors = [main_author_name] + coauthor_names
        else:
            coauthor_names = []
            authors = [main_author_name]

        # Yukarıdaki verileri graph'a ekle
        graph.add_paper(
            paper_title=paper_title,
            doi=doi,
            main_author_name=main_author_name,
            author_position=author_position,
            coauthors=coauthors,
            all_authors_info=all_authors_info,
            authors=authors
        )

    return graph


# -----------------------------------------------------
# Dairesel Yerleşim Fonksiyonu
# -----------------------------------------------------
def create_custom_circular_layout(graph):
    """
    Belirli bir dairesel yerleşim algoritması uygular:
      - Baş yazarları (author_position=1) dış bir çembere
      - Co-author'ları da daha içteki bir çembere yerleştirir.
    Dönüş: positions dict -> { 'mehmet': (x,y), 'ali veli': (x2,y2), ... }
    """
    positions = {}
    placed_nodes = set()

    # 1) Baş yazarları listele
    main_authors = [node for node in graph.nodes.values() if node.author_position == 1]
    if not main_authors:
        # Eğer baş yazar yoksa, tüm yazarları birinci halka gibi kabul edelim
        main_authors = list(graph.nodes.values())

    main_authors.sort(key=lambda x: x.name)
    N = len(main_authors)
    if N == 0:
        return positions  # Boşsa direkt geri dön

    max_r = 300  # Alt dairelerin max yarıçapı
    # Alt daire yarıçapları (başyazarın makale sayısına göre)
    subcircle_sizes = []
    for ma in main_authors:
        r_i = 50 + 20 * len(ma.papers)  # Makale sayısı arttıkça çember genişlet
        if r_i > max_r:
            r_i = max_r
        subcircle_sizes.append(r_i)

    avg_subcircle = sum(subcircle_sizes) / len(subcircle_sizes) if subcircle_sizes else 50

    # Büyük daire yarıçapı (tüm baş yazarlar için)
    BIG_RADIUS = 600 + 20 * N + 2 * avg_subcircle
    if BIG_RADIUS < 300:
        BIG_RADIUS = 300

    # Ekranda grafiği biraz ortalamak için x, y değerlerini
    BIG_CENTER_X = 400
    BIG_CENTER_Y = 400

    angle_step_main = 360 / N  # Baş yazarlar arasındaki açı farkı
    current_angle = 0

    # Her baş yazarın konumunu hesapla
    for ma_node in main_authors:
        r_i = 50 + 20 * len(ma_node.papers)
        if r_i > max_r:
            r_i = max_r

        rad_main = math.radians(current_angle)
        center_x = BIG_CENTER_X + (BIG_RADIUS - r_i) * math.cos(rad_main)
        center_y = BIG_CENTER_Y + (BIG_RADIUS - r_i) * math.sin(rad_main)

        positions[ma_node.name.lower()] = (center_x, center_y)
        placed_nodes.add(ma_node.name.lower())

        # Alt dairede coauthor'ları yerleştirelim
        co_keys = list(ma_node.edges.keys())
        co_count = len(co_keys)
        if co_count > 0:
            subcircle_radius = r_i - 50
            angle_step_small = 360 / co_count
            small_angle = 0

            for co_lower in co_keys:
                # Eğer coauthor zaten yerleştirilmişse (örneğin başka baş yazarın altındaysa) atla
                if co_lower in placed_nodes:
                    small_angle += angle_step_small
                    continue
                co_node = graph.nodes.get(co_lower)
                if not co_node:
                    small_angle += angle_step_small
                    continue

                rad_small = math.radians(small_angle)
                co_x = center_x + subcircle_radius * math.cos(rad_small)
                co_y = center_y + subcircle_radius * math.sin(rad_small)

                positions[co_lower] = (co_x, co_y)
                placed_nodes.add(co_lower)
                small_angle += angle_step_small

        current_angle += angle_step_main

    # Örnek bir çakışma engelleme yöntemi (basit)
    for node, position in positions.items():
        x, y = position
        for other_node, other_position in positions.items():
            if node != other_node and (x, y) == other_position:
                new_x = BIG_CENTER_X + (BIG_RADIUS // 2) * math.cos(math.radians(90))
                new_y = BIG_CENTER_Y + (BIG_RADIUS // 2) * math.sin(math.radians(90))
                positions[node] = (new_x, new_y)

    return positions


# -----------------------------------------------------
# Cytoscape Elemanlarını Oluşturma
# -----------------------------------------------------
def create_cytoscape_elements(graph, paper_counts, average_papers, positions):
    """
    Dash Cytoscape'de kullanılmak üzere 'elements' listesini oluşturur.
    Hem düğümleri (node) hem de kenarları (edge) ekler.
    Düğüm boyutu ve rengi, yazarın makale sayısına göre ayarlanır.
    Kenar kalınlığı ve rengi, 'weight' değerine göre ayarlanır.
    """
    elements = []

    # En büyük edge weight'i bulalım
    max_edge_weight = max(
        (edge_info['weight'] for node in graph.nodes.values() for neighbor, edge_info in node.edges.items()),
        default=1
    )

    # --------------------------
    # 1) Düğümleri ekle
    # --------------------------
    for name_lower, node in graph.nodes.items():
        node_paper_count = paper_counts.get(node.name, 0)
        size_factor = 10  # Bir ölçek değeri
        color = '#87CEFA'  # Normal renk (açık mavi)
        size = 40  # Varsayılan düğüm boyutu

        # Ortalama makale sayısının %20 fazlasından büyükse daha büyük ve farklı renk
        if node_paper_count > average_papers * 1.2:
            size = 60
            color = '#1E90FF'  # Koyu mavi
        # Ortalama makale sayısının %20 altındaysa daha küçük
        elif node_paper_count < average_papers * 0.8:
            size = 30
            color = '#ADD8E6'  # Daha açık mavi

        # positions sözlüğünde bulduğumuz (x, y)
        pos = positions.get(name_lower, (0, 0))

        elements.append({
            'data': {'id': node.name},
            'position': {'x': pos[0], 'y': pos[1]},
            'style': {
                'width': size,
                'height': size,
                'background-color': color
            }
        })

    # --------------------------
    # 2) Kenarları ekle
    # --------------------------
    added_edges = set()
    for node in graph.nodes.values():
        source = node.name
        for neighbor in node.edges.keys():
            if neighbor not in graph.nodes:
                continue
            target = graph.nodes[neighbor].name
            edge = tuple(sorted((source, target)))
            if edge not in added_edges:
                added_edges.add(edge)
                edge_weight = node.edges[neighbor]['weight']

                # Kenar kalınlığı, en fazla 6 olacak şekilde ayar
                edge_thickness = min(2 + edge_weight * 0.5, 6)
                # Renk: ağırlık arttıkça biraz daha koyulaşsın (örnek bir yaklaşım)
                edge_color = f"rgb({255 - int(50 * edge_weight)}, {255 - int(30 * edge_weight)}, {255 - int(10 * edge_weight)})"

                elements.append({
                    'data': {
                        'source': source,
                        'target': target,
                        'weight': edge_weight
                    },
                    'style': {
                        'line-color': edge_color,
                        'width': edge_thickness,
                        'target-arrow-color': edge_color,
                        'target-arrow-shape': 'triangle',
                    }
                })

    return elements


# -----------------------------------------------------
# BST Fonksiyonları
# -----------------------------------------------------
def bst_insert(root, cname, w):
    """
    Binary Search Tree'ye (BST) yeni bir düğüm ekleyen fonksiyon.
    cname: yazarın ismi, w: weight (kenar ağırlığı).
    """
    if root is None:
        return {'cname': cname, 'weight': w, 'left': None, 'right': None}
    if w < root['weight']:
        root['left'] = bst_insert(root['left'], cname, w)
    else:
        root['right'] = bst_insert(root['right'], cname, w)
    return root

def bst_remove(root, cname):
    """
    BST'den belirtilen 'cname' yazarını siler.
    """
    if root is None:
        return root
    if cname < root['cname']:
        root['left'] = bst_remove(root['left'], cname)
    elif cname > root['cname']:
        root['right'] = bst_remove(root['right'], cname)
    else:
        # Silinecek düğüm bulundu
        if root['left'] is None:
            temp = root['right']
            root = None
            return temp
        elif root['right'] is None:
            temp = root['left']
            root = None
            return temp
        # İki çocuğu varsa, en küçük değerli sağ alt düğümü getir
        temp = get_min_value_node(root['right'])
        root['cname'] = temp['cname']
        root['weight'] = temp['weight']
        root['right'] = bst_remove(root['right'], temp['cname'])
    return root

def get_min_value_node(node):
    """
    BST'de en küçük değerli düğümü bulan yardımcı fonksiyon.
    Soldan ilerleyerek bulur.
    """
    current = node
    while current['left'] is not None:
        current = current['left']
    return current

def bst_to_text(root):
    """
    BST'yi metin formatında seviyeler halinde görüntülemek için bir BFS yaklaşımı.
    Yoksa "BST ağacı boş." döndürür.
    """
    if not root:
        return "BST ağacı boş."

    queue = [(root, 0)]
    levels = {}

    while queue:
        node, depth = queue.pop(0)
        levels.setdefault(depth, []).append(f"{node['cname']}({node['weight']})")

        if node['left']:
            queue.append((node['left'], depth+1))
        if node['right']:
            queue.append((node['right'], depth+1))

    text_lines = []
    for d in sorted(levels.keys()):
        line = f"Seviye {d}: " + ", ".join(levels[d])
        text_lines.append(line)
    return "\n".join(text_lines)


# -----------------------------------------------------
# Dijkstra (En Kısa Yol) Fonksiyonu
# -----------------------------------------------------
def dijkstra(graph, start, target):
    """
    Basit bir Dijkstra uygulaması.
    graph: {'ali': {'veli': 2, 'ayşe': 3}, ...} şeklinde adjacency list.
    start, target: başlangıç ve hedef düğüm isimleri.
    Dönüş: (path, distance)
      path: start -> target en kısa yolu (düğümlerin listesi)
      distance: top. mesafe (weight)
    """
    distances = {node: float('inf') for node in graph}
    distances[start] = 0
    previous_nodes = {node: None for node in graph}
    unvisited_nodes = list(graph)

    while unvisited_nodes:
        # Henüz ziyaret edilmemiş düğümler arasında en kısa mesafeli olanı bul
        min_node = None
        for node in unvisited_nodes:
            if min_node is None:
                min_node = node
            elif distances[node] < distances[min_node]:
                min_node = node

        for neighbour, weight in graph[min_node].items():
            if neighbour in unvisited_nodes:
                alternative_route = distances[min_node] + weight
                if alternative_route < distances[neighbour]:
                    distances[neighbour] = alternative_route
                    previous_nodes[neighbour] = min_node

        unvisited_nodes.remove(min_node)

    path = []
    current_node = target
    while previous_nodes[current_node] is not None:
        path.append(current_node)
        current_node = previous_nodes[current_node]
    path.append(start)
    path.reverse()

    return path, distances[target]


# -----------------------------------------------------
# Dijkstra Shortest Path All (İster 4)
# -----------------------------------------------------
def dijkstra_shortest_path_all(graphdata, start):
    """
    Bir yazarın (start) diğer tüm yazarlara olan en kısa yollarını hesaplar.
    graphdata: GraphData nesnesi.
    start: başlangıç yazarın adı (string).
    Dönüş: (all_paths, queue_steps)
      all_paths: { 'mehmet': [start, ..., 'mehmet'], ... }
      queue_steps: her adımda kuyruk durumunu gösteren string listesi (debug amaçlı).
    """
    if start not in graphdata.nodes:
        return None, []

    adjacency = {}
    for name_lower, node in graphdata.nodes.items():
        adjacency[node.name] = node.edges

    distances = {node.name: math.inf for node in graphdata.nodes.values()}
    previous = {node.name: None for node in graphdata.nodes.values()}
    distances[start] = 0

    queue_steps = []
    unvisited = set(adjacency.keys())

    while unvisited:
        current = min(unvisited, key=lambda x: distances[x])
        unvisited.remove(current)
        queue_steps.append(f"Kuyruk: {sorted(unvisited)}\nŞu anki düğüm: {current}, Mesafe: {distances[current]}")

        if distances[current] == math.inf:
            break

        for neighbor, edge_info in adjacency[current].items():
            alt = distances[current] + edge_info['weight']
            if alt < distances[neighbor]:
                distances[neighbor] = alt
                previous[neighbor] = current

    all_paths = {}
    for target in adjacency.keys():
        if target == start:
            continue
        path = []
        c = target
        while c is not None:
            path.insert(0, c)
            c = previous[c]
        if distances[target] != math.inf:
            all_paths[target] = path
        else:
            all_paths[target] = None

    return all_paths, queue_steps


# -----------------------------------------------------
# Priority Queue (İster 2)
# -----------------------------------------------------
def create_priority_queue_for_author(graphdata, author_name):
    """
    Bir yazarın (A) coauthor'larını, 'weight' değerine göre
    büyükten küçüğe sıralayarak (bir nevi priority queue),
    geri döndürür.
    """
    key = author_name.lower()
    if key not in graphdata.nodes:
        return None, "Girilen A yazarı bulunamadı."

    a_node = graphdata.nodes.get(key, None)
    if not a_node:
        return None, "A yazarı bulunamadı."

    coauthor_names = list(a_node.edges.keys())
    if not coauthor_names:
        return None, "A yazarının işbirliği yaptığı yazar bulunmuyor."

    coauthors_info = []
    for cname in coauthor_names:
        c_node = graphdata.nodes.get(cname.lower(), None)
        if c_node:
            # Yazar A ile arasındaki kenar ağırlığını al
            weight = c_node.edges[key]['weight']
            coauthors_info.append({'author': c_node.name, 'weight': weight})

    # Coauthor'ları 'weight' değerine göre (büyükten küçüğe) sırala
    coauthors_info.sort(key=lambda x: x['weight'], reverse=True)
    coauthor_count = len(coauthors_info)

    return coauthors_info, coauthor_count


# -----------------------------------------------------
# Dash Uygulamasını Çalıştırma
# -----------------------------------------------------
def run_dash(graph, cy_elements, paper_counts, average_papers, main_author_names, reverse_node_id_map):
    """
    Dash üzerinden bir web arayüzü kurar.
    Grafiği (Cytoscape), butonları ve metin alanlarını ayarlar.
    Callback fonksiyonlarıyla etkileşimi yönetir.
    """
    app = dash.Dash(__name__)
    app.title = "Yazar İşbirliği Ağı"

    # Global değişkenler. Diğer callback fonksiyonlarından erişebilmek için.
    run_dash.global_coauthors_info = []
    run_dash.bst_root = None

    # Cytoscape stylesheet: Düğümlerin, kenarların görünümü için CSS benzeri kurallar
    cyto_stylesheet = [
        {
            'selector': 'node',
            'style': {
                'font-size': '12px',
                'text-valign': 'center',
                'text-halign': 'center',
                'color': 'black',
                'background-color': '#1E90FF',
                'border-color': '#ffffff',
                'border-width': 1,
                'shape': 'ellipse',
                'transition-property': 'background-color, border-color',
                'transition-duration': '0.5s'
            }
        },
        {
            'selector': 'edge',
            'style': {
                'width': 2,
                'line-color': '#ADD8E6',
                'curve-style': 'bezier',
                'target-arrow-shape': 'triangle',
                'target-arrow-color': '#ADD8E6',
                'source-arrow-shape': 'none',
                'transition-property': 'line-color',
                'transition-duration': '0.5s'
            }
        },
        {
            # highlight class'ına sahip node veya edge daha kalın ve kırmızı görünsün
            'selector': '.highlighted',
            'style': {
                'background-color': '#FF4136',
                'line-color': '#FF4136',
                'target-arrow-color': '#FF4136',
                'width': 4
            }
        }
    ]

    button_style = {
        "marginBottom": "5px",
        "width": "100%",
        "padding": "10px",
        "backgroundColor": "#66CCFF",
        "color": "black",
        "border": "none",
        "borderRadius": "5px",
        "cursor": "pointer",
        "fontWeight": "bold"
    }

    # -------------------------
    # Dash Layout
    # -------------------------
    app.layout = html.Div([
        # Store bileşenleri: Dash'te verileri saklamak için.
        dcc.Store(id='selected-node', storage_type='memory'),
        dcc.Store(id='zoom-nodes', data=[]),  # Zoomlanacak düğümleri tutar
        html.Div(id='zoom-output', style={'display': 'none'}),  # Gizli div

        html.Div([
            # SOL TARAF: ÇIKTI EKRANI
            html.Div([
                html.H3("ÇIKTI EKRANI"),
                html.Div(id="result-container", style={
                    "whiteSpace": "pre-line",
                    "height": "800px",
                    "border": "1px solid #ccc",
                    "padding": "10px",
                    "overflowY": "auto",
                    "backgroundColor": "#004080",
                    "borderRadius": "5px",
                    "color": "white"
                }),
            ], style={
                "width": "25%",
                "display": "flex",
                "flexDirection": "column",
                "padding": "10px",
                "boxSizing": "border-box"
            }),

            # ORTA: Network Graph (Cytoscape)
            html.Div([
                cyto.Cytoscape(
                    id='network-graph',
                    elements=cy_elements,  # create_cytoscape_elements ile oluşturulan elemanlar
                    style={'width': '100%', 'height': '800px'},
                    layout={'name': 'preset'},  # 'preset' -> biz elle pozisyon veriyoruz (positions)
                    stylesheet=cyto_stylesheet,
                    userZoomingEnabled=True,
                    userPanningEnabled=True,
                    boxSelectionEnabled=False,
                    autounselectify=True
                )
            ], style={
                "width": "50%",
                "display": "flex",
                "flexDirection": "column",
                "padding": "10px",
                "boxSizing": "border-box",
                "backgroundColor": "#003366",
                "borderRadius": "5px",
                "height": "800px"
            }),

            # SAĞ TARAF: İSTERLER DÜĞMELERİ
            html.Div([
                html.H3("İSTERLER"),
                html.Div([
                    html.Button("1. En Kısa Yol Bulma", id="ister1-btn", n_clicks=0, style=button_style),
                    html.Div([
                        html.Label("A Yazarı İsmi:"),
                        dcc.Input(id='authorA-input', type='text', placeholder='Author A İsmi',
                                  style={"marginBottom": "10px", "width": "100%"}),
                        html.Label("B Yazarı İsmi:"),
                        dcc.Input(id='authorB-input', type='text', placeholder='Author B İsmi',
                                  style={"marginBottom": "10px", "width": "100%"})
                    ], style={"marginBottom": "15px"})
                ]),

                html.Div([
                    html.Button("2. Kuyruk Oluşturma", id="ister2-btn", n_clicks=0, style=button_style),
                    html.Div([
                        html.Label("A Yazarı İsmi:"),
                        dcc.Input(id='authorA-input-ister2', type='text', placeholder='Author A İsmi',
                                  style={"marginBottom": "10px", "width": "100%"})
                    ], style={"marginBottom": "15px"})
                ]),

                html.Div([
                    html.Button("3. Kuyruktaki Yazarlarla BST Oluşturma", id="ister3-btn", n_clicks=0, style=button_style),
                    html.Div([
                        html.Label("BST'den Silinecek Yazar İsmi:"),
                        dcc.Input(id='bst-remove-name', type='text', placeholder='Silinecek Yazar İsmi',
                                  style={"marginBottom": "10px", "width": "100%"})
                    ], style={"marginBottom": "15px"})
                ]),

                html.Div([
                    html.Button("4. Kısa Yolların Hesaplanması", id="ister4-btn", n_clicks=0, style=button_style),
                    html.Div([
                        html.Label("A Yazarı İsmi:"),
                        dcc.Input(id='authorA-input-ister4', type='text', placeholder='Author A İsmi',
                                  style={"marginBottom": "10px", "width": "100%"})
                    ], style={"marginBottom": "15px"})
                ]),

                html.Div([
                    html.Button("5. İşbirliği Sayısının Hesaplanması", id="ister5-btn", n_clicks=0, style=button_style),
                    html.Div([
                        html.Label("A Yazarı İsmi:"),
                        dcc.Input(id='authorA-input-ister5', type='text', placeholder='Author A İsmi',
                                  style={"marginBottom": "10px", "width": "100%"})
                    ], style={"marginBottom": "15px"})
                ]),

                html.Div([
                    html.Button("6. En Çok İşbirliği Yapan Yazar", id="ister6-btn", n_clicks=0, style=button_style)
                ], style={"marginBottom": "15px"}),

                html.Div([
                    html.Button("7. En Uzun Yol Bulma", id="ister7-btn", n_clicks=0, style=button_style),
                    html.Div([
                        html.Label("Başlangıç Yazarının İsmi:"),
                        dcc.Input(id='authorA-input-ister7', type='text', placeholder='Başlangıç Yazarının İsmi',
                                  style={"marginBottom": "10px", "width": "100%"})
                    ], style={"marginBottom": "15px"})
                ]),
            ], style={
                "width": "25%",
                "display": "flex",
                "flexDirection": "column",
                "padding": "10px",
                "boxSizing": "border-box",
                "overflowY": "auto",
                "backgroundColor": "#004080",
                "borderRadius": "5px",
                "height": "800px",
                "color": "white"
            })

        ], style={
            'display': 'flex',
            'flexDirection': 'row',
            'justifyContent': 'space-between',
            'alignItems': 'stretch'
        }),

    ], style={
        'fontFamily': 'Arial, sans-serif',
        'backgroundColor': '#001f3f',
        'padding': '20px',
        'color': 'white'
    })

    # -----------------------------------------------------
    # Clientside Callback: Zoomlama İşlemi (Tarayıcı tarafında)
    # -----------------------------------------------------
    # Bu callback JavaScript tarafında çalışır, 'zoom-nodes' store'da tutulan isimlere göre 
    # graph üzerinde fit(zoom) işlemi uygular.
    app.clientside_callback(
        """
        function(zoom_nodes) {
            if (!zoom_nodes || zoom_nodes.length === 0) {
                return "";
            }
            const cyo = document.getElementById('network-graph');
            if (cyo && cyo.cy) {
                const cy_instance = cyo.cy;
                const selector = zoom_nodes.map(name => [id = "${name}"]).join(', ');
                cy_instance.fit(cy_instance.$(selector), { padding: 50 });
            }
            return "";
        }
        """,
        Output('zoom-output', 'children'),
        Input('zoom-nodes', 'data')
    )

    # -----------------------------------------------------
    # Python Callback: Bütün buton tıklamaları vb.
    # -----------------------------------------------------
    @app.callback(
        [Output('result-container', 'children'),
         Output('network-graph', 'elements'),
         Output('selected-node', 'data'),
         Output('zoom-nodes', 'data')],
        [
            Input('ister1-btn', 'n_clicks'),
            Input('ister2-btn', 'n_clicks'),
            Input('ister3-btn', 'n_clicks'),
            Input('ister4-btn', 'n_clicks'),
            Input('ister5-btn', 'n_clicks'),
            Input('ister6-btn', 'n_clicks'),
            Input('ister7-btn', 'n_clicks'),
            Input('network-graph', 'tapNodeData')
        ],
        [
            State('authorA-input', 'value'),
            State('authorB-input', 'value'),
            State('bst-remove-name', 'value'),
            State('selected-node', 'data'),
            State('network-graph', 'elements'),
            State('authorA-input-ister2', 'value'),
            State('authorA-input-ister4', 'value'),
            State('authorA-input-ister5', 'value'),
            State('authorA-input-ister7', 'value'),
        ]
    )
    def update_output(ister1, ister2, ister3, ister4, ister5, ister6, ister7, tapNodeData,
                      authorA, authorB, bst_remove_name, selected_node, current_elements,
                      authorA_ister2, authorA_ister4, authorA_ister5, authorA_ister7):
        """
        Callback fonksiyonu:
          - Her buton tıklaması (ister1..7-btn) ve
          - Graf üzerindeki bir düğüme tıklama (tapNodeData)
          olaylarını yönetir.
        Dönen değerler: 
          - result-container içeriği
          - network-graph'ın elements listesi
          - seçili düğüm bilgisi
          - zoom-nodes (list of node names for zoom)
        """
        import dash
        triggered = dash.callback_context.triggered
        if not triggered:
            # Hiçbir tetikleme olmadıysa varsayılan değerleri döndür
            return "", cy_elements, selected_node, dash.no_update

        triggered_id = triggered[0]['prop_id'].split('.')[0]
        result_content = ""
        new_elements = current_elements[:]
        new_selected_node = selected_node
        zoom_nodes = dash.no_update  # Varsayılan olarak değişmeyecek

        # Her yeni istekte önceki highlight'ları temizliyoruz
        for el in new_elements:
            if 'classes' in el:
                el['classes'] = []

        # Eğer graf düğümüne tıklandıysa (tapNodeData)
        if triggered_id == 'network-graph' and tapNodeData:
            node_name = tapNodeData['id']
            # Toggle mantığı: Aynı düğüme tekrar tıklarsak seçimi kaldırırız
            if selected_node == node_name:
                new_selected_node = None
                for element in new_elements:
                    if element.get('data', {}).get('id') == node_name and 'classes' in element:
                        element['classes'] = []
            else:
                new_selected_node = node_name
                for element in new_elements:
                    if 'classes' in element:
                        element['classes'] = []
                for element in new_elements:
                    if element.get('data', {}).get('id') == node_name:
                        element['classes'] = element.get('classes', [])
                        element['classes'].append('highlighted')

            # Yazarın bilgilerini ekranda göstermek için
            node = graph.nodes.get(node_name.lower(), None)
            if node:
                node_info = {
                    "Adı": node.name,
                    "Makale Sayısı": len(node.papers)
                }
                coauthors = {}
                for paper in node.papers:
                    paper_title, doi, ap, main_author, authors_in_paper = paper
                    co_names = [a for a in authors_in_paper if a.lower() != node.name.lower()]
                    for c in co_names:
                        c_cl = clean_text(c)
                        if c_cl not in coauthors:
                            coauthors[c_cl] = []
                        coauthors[c_cl].append(paper_title)

                if coauthors:
                    co_list = []
                    for co, papers_ in coauthors.items():
                        co_display = next((n for n in graph.nodes.keys() if n.lower() == co), co)
                        co_list.append(html.Li([
                            html.Strong(co_display),
                            html.Ul([html.Li(p) for p in papers_])
                        ]))
                    coauthors_display = html.Ul(co_list)
                else:
                    coauthors_display = html.P("Ortak çalışılan yazar bulunmamaktadır.")

                author_details = html.Div([
                    html.H4(f"Yazar: {node_info['Adı']}"),
                    html.P(f"Makale Sayısı: {node_info['Makale Sayısı']}"),
                    html.H5("Ortak Çalışılan Yazarlar ve Makaleler:"),
                    coauthors_display
                ])
                return author_details, new_elements, new_selected_node, dash.no_update
            else:
                return "Hata: Yazar bilgisi bulunamadı.", new_elements, new_selected_node, dash.no_update

        # -----------------------------------------------------
        # İSTER 1: En Kısa Yol (A-B Arası)
        # -----------------------------------------------------
        elif triggered_id == 'ister1-btn':
            if not authorA or not authorB:
                return "Lütfen A ve B yazar isimlerini giriniz.", cy_elements, selected_node, dash.no_update

            authorA_name = clean_text(authorA)
            authorB_name = clean_text(authorB)
            if authorA_name not in graph.nodes or authorB_name not in graph.nodes:
                return "Lütfen geçerli yazar isimleri giriniz.", cy_elements, selected_node, dash.no_update

            # NetworkX Graph oluştur (alternatif olarak, kendimiz adjacency de kullanabiliriz)
            G = nx.Graph()
            for node in graph.nodes.values():
                for neighbor, edge_info in node.edges.items():
                    G.add_edge(node.name, graph.nodes[neighbor].name, weight=edge_info['weight'])

            # A'dan B'ye tüm basit yolları bul
            try:
                all_paths = list(nx.all_simple_paths(G, source=graph.nodes[authorA_name].name,
                                                     target=graph.nodes[authorB_name].name))
            except nx.NetworkXNoPath:
                all_paths = []

            if not all_paths:
                result_content = html.Div([
                    html.Div("A ile B arasında bir yol bulunamadı.", style={
                        'border': '2px solid red',
                        'padding': '10px',
                        'borderRadius': '5px',
                        'backgroundColor': '#ffe6e6',
                        'color': 'red'
                    })
                ])
                return result_content, cy_elements, selected_node, dash.no_update

            # En kısa yol = minimum düğüm sayısına sahip yol
            shortest_length = min(len(path) for path in all_paths)
            shortest_paths = [path for path in all_paths if len(path) == shortest_length]

            # Bütün yolları ekrana kutucuk halinde göster
            path_boxes = []
            for idx, path in enumerate(all_paths, 1):
                style = {
                    'border': '2px solid #1E90FF',
                    'padding': '10px',
                    'marginBottom': '10px',
                    'borderRadius': '5px',
                    'backgroundColor': '#00509E',
                    'color': 'white'
                }
                # En kısa yollardan biriyse rengi değiştir
                if path in shortest_paths:
                    style['border'] = '2px solid #FFD700'
                    style['backgroundColor'] = '#87CEEB'
                path_boxes.append(html.Div([
                    html.H5(f"Yol {idx}:", style={'marginBottom': '5px'}),
                    html.P(" -> ".join(path)),
                    html.P("(En Kısa Yol)" if path in shortest_paths else "")
                ], style=style))

            # Herhangi bir en kısa yolun ilkini alıyoruz (zooma göstermek için)
            shortest_path = shortest_paths[0]

            # En kısa yolu grafikte highlight
            for element in new_elements:
                if 'source' in element.get('data', {}):
                    src = element['data']['source']
                    tgt = element['data']['target']
                    if src in shortest_path and tgt in shortest_path:
                        idx_src = shortest_path.index(src)
                        idx_tgt = shortest_path.index(tgt)
                        if abs(idx_src - idx_tgt) == 1:
                            element['classes'] = element.get('classes', [])
                            element['classes'].append('highlighted')
                elif 'id' in element.get('data', {}):
                    if element['data']['id'] in shortest_path:
                        element['classes'] = element.get('classes', [])
                        element['classes'].append('highlighted')

            # Zoomlamada gösterilecek düğümler: en kısa yol üzerindeki tüm düğümler
            zoom_nodes = shortest_path

            result_content = html.Div([
                html.H4("A ile B arasındaki yollar:"),
                *path_boxes
            ])

            return result_content, new_elements, selected_node, zoom_nodes

        # -----------------------------------------------------
        # İSTER 2: Kuyruk Oluşturma
        # -----------------------------------------------------
        elif triggered_id == 'ister2-btn':
            if not authorA_ister2:
                return "Lütfen A yazarının ismini giriniz.", cy_elements, selected_node, dash.no_update

            authorA_name = clean_text(authorA_ister2)
            if authorA_name not in graph.nodes:
                return "Girilen A yazarı bulunamadı.", cy_elements, selected_node, dash.no_update

            coauthors_info, coauthor_count = create_priority_queue_for_author(graph, graph.nodes[authorA_name].name)
            if coauthors_info is None:
                # coauthor_count burada hata mesajı içerebilir
                return coauthor_count, cy_elements, selected_node, dash.no_update

            # Kuyruk bilgilerini global saklıyoruz
            run_dash.global_coauthors_info = coauthors_info

            # Ana yazarın box'ı
            main_author = graph.nodes[authorA_name]
            main_author_display = html.Div([
                html.Div([
                    html.H4(f"{main_author.name}"),
                    html.P(f"İşbirliği Yapılan Yazar Sayısı: {coauthor_count}")
                ], style={
                    'border': '2px solid #1E90FF',
                    'borderRadius': '10px',
                    'padding': '10px',
                    'marginBottom': '20px',
                    'backgroundColor': '#003366'
                })
            ])

            # Coauthor bilgilerini alt alta küçük kutucuklar olarak
            coauthor_boxes = []
            for co in coauthors_info:
                coauthor_boxes.append(
                    html.Div([
                        html.H5(co['author']),
                        html.P(f"Kenar Ağırlığı: {co['weight']}")
                    ], style={
                        'border': '1px solid #87CEFA',
                        'borderRadius': '10px',
                        'padding': '10px',
                        'margin': '5px 0',
                        'backgroundColor': '#00509E',
                        'color': 'white',
                        'width': '100%'
                    })
                )

            coauthors_display = html.Div(coauthor_boxes, style={'display': 'block'})

            result_content = html.Div([
                main_author_display,
                html.H4("İşbirliği Yapılan Yazarlar:"),
                coauthors_display
            ])

            # Zoom: Ana yazar + coauthor'lar
            zoom_nodes = [main_author.name] + [co['author'] for co in coauthors_info]

            return result_content, cy_elements, selected_node, zoom_nodes

        # -----------------------------------------------------
        # İSTER 3: BST Oluşturma & Eleman Silme
        # -----------------------------------------------------
        elif triggered_id == 'ister3-btn':
            coauthors_info = getattr(run_dash, 'global_coauthors_info', None)
            if not coauthors_info:
                return "Önce 2. İster'i gerçekleştirin, kuyruk oluşturun.", cy_elements, selected_node, dash.no_update

            bst_root = run_dash.bst_root
            # Kuyruktaki yazarları BST'ye ekle
            for co in coauthors_info:
                bst_root = bst_insert(bst_root, co['author'], co['weight'])

            if not bst_remove_name:
                return "Lütfen BST'den çıkarılacak yazar ismini giriniz.", cy_elements, selected_node, dash.no_update

            cname_remove = clean_text(bst_remove_name)
            old_bst_root = bst_root
            bst_root = bst_remove(bst_root, cname_remove)

            if bst_root == old_bst_root:
                result_content = f"Belirtilen yazar ({cname_remove.title()}) BST'de bulunamadı.\n"
            else:
                result_content = f"Yazar ({cname_remove.title()}) BST'den çıkarıldı.\n"

            run_dash.bst_root = bst_root

            # BST'nin güncel durumunu yazı olarak al
            bst_text = bst_to_text(bst_root)
            result_content += "\nBST Güncel Durumu:\n" + bst_text

            return result_content, cy_elements, selected_node, dash.no_update

        # -----------------------------------------------------
        # İSTER 4: Tüm Kısa Yolların Hesaplanması
        # -----------------------------------------------------
        elif triggered_id == 'ister4-btn':
            if not authorA_ister4:
                return "Lütfen A yazarının ismini giriniz.", cy_elements, selected_node, dash.no_update
            authorA_name = clean_text(authorA_ister4)

            if authorA_name not in graph.nodes:
                return "Girilen A yazarı bulunamadı.", cy_elements, selected_node, dash.no_update

            all_paths, queue_steps = dijkstra_shortest_path_all(graph, graph.nodes[authorA_name].name)
            if all_paths is None:
                return "Girilen A yazarı grafikte bulunmuyor.", cy_elements, selected_node, dash.no_update

            result_content = f"A yazarının ({graph.nodes[authorA_name].name}) diğer yazarlarla en kısa yolları:\n\n"

            path_boxes = []
            for target, path in all_paths.items():
                if path:
                    path_boxes.append(html.Div([
                        html.H5(f"Yol: {graph.nodes[authorA_name].name} -> {target}"),
                        html.P(" -> ".join(path)),
                        html.P("(Yol bulundu)")
                    ], style={
                        'border': '2px solid #1E90FF',
                        'padding': '10px',
                        'marginBottom': '10px',
                        'borderRadius': '5px',
                        'backgroundColor': '#00509E',
                        'color': 'white'
                    }))

            if not path_boxes:
                result_content += "Hiçbir yazarla yol bulunamadı.\n"
            else:
                result_content += "\nYollar başarıyla hesaplandı."

            result_content = html.Div([result_content] + path_boxes)

            return result_content, cy_elements, selected_node, dash.no_update

        # -----------------------------------------------------
        # İSTER 5: İşbirliği Sayısı (Belirli bir yazarın kaç coauthor'u var)
        # -----------------------------------------------------
        elif triggered_id == 'ister5-btn':
            if not authorA_ister5:
                return "Lütfen A yazarının ismini giriniz.", cy_elements, selected_node, dash.no_update

            authorA_name = clean_text(authorA_ister5)
            if authorA_name not in graph.nodes:
                return f"A yazarı ({authorA_ister5.title()}) grafikte bulunmuyor.", cy_elements, selected_node, dash.no_update

            node_ = graph.nodes.get(authorA_name, None)
            if node_:
                co_count = len(node_.edges)
                result_content += f"A yazarının ({node_.name}) işbirliği yaptığı toplam yazar sayısı: {co_count}"
                return result_content, cy_elements, selected_node, dash.no_update
            else:
                return "Hata: Yazar bilgisi bulunamadı.", cy_elements, selected_node, dash.no_update

        # -----------------------------------------------------
        # İSTER 6: En Çok İşbirliği Yapan Yazar
        # -----------------------------------------------------
        elif triggered_id == 'ister6-btn':
            if not graph.nodes:
                return "Graf boş. En çok işbirliği yapan yazar yok.", cy_elements, selected_node, dash.no_update

            max_collab = -1
            top_authors = []
            for node_ in graph.nodes.values():
                collab_count = len(node_.edges)
                if collab_count > max_collab:
                    max_collab = collab_count
                    top_authors = [node_.name]
                elif collab_count == max_collab:
                    top_authors.append(node_.name)

            if top_authors:
                result_content += f"En çok işbirliği yapan yazar(lar), işbirliği sayısı = {max_collab}\n"
                for ta in top_authors:
                    result_content += f"- {ta}\n"
            else:
                result_content += "En çok işbirliği yapan yazar bulunamadı."

            return result_content, cy_elements, selected_node, dash.no_update

        # -----------------------------------------------------
        # İSTER 7: En Uzun Yol
        # -----------------------------------------------------
        elif triggered_id == 'ister7-btn':
            if not authorA_ister7:
                return "Lütfen başlangıç yazarının ismini giriniz.", cy_elements, selected_node, dash.no_update

            authorA_name = clean_text(authorA_ister7)
            if authorA_name not in graph.nodes:
                return f"Yazar ({authorA_ister7}) grafikte bulunmuyor.", cy_elements, selected_node, dash.no_update

            # DFS kullanarak en uzun yolu arama
            def dfs_longest_path(gdata, current, visited, path, longest):
                visited.add(current)
                path.append(current)
                if len(path) > len(longest[0]):
                    longest[0] = list(path)
                for neighbor in gdata.nodes[current.lower()].edges.keys():
                    neigh_name = gdata.nodes[neighbor.lower()].name
                    if neigh_name not in visited:
                        dfs_longest_path(gdata, neigh_name, visited, path, longest)
                path.pop()
                visited.remove(current)

            longest_path = [[]]
            dfs_longest_path(graph, graph.nodes[authorA_name].name, set(), [], longest_path)

            if longest_path[0]:
                result_content += f"En uzun yol ({len(longest_path[0])} düğüm): {' -> '.join(longest_path[0])}\n"
                path = longest_path[0]
                # Bu yolu grafikte highlight
                for element in new_elements:
                    if 'source' in element.get('data', {}):
                        src = element['data']['source']
                        tgt = element['data']['target']
                        if src in path and tgt in path:
                            idx_src = path.index(src)
                            idx_tgt = path.index(tgt)
                            if abs(idx_src - idx_tgt) == 1:
                                element['classes'] = element.get('classes', [])
                                element['classes'].append('highlighted')
                    elif 'id' in element.get('data', {}):
                        if element['data']['id'] in path:
                            element['classes'] = element.get('classes', [])
                            element['classes'].append('highlighted')
            else:
                result_content += "En uzun yol bulunamadı.\n"

            return result_content, new_elements, selected_node, dash.no_update

        # Eğer hiçbir koşul sağlanmadıysa
        return "", cy_elements, selected_node, dash.no_update

    # Dash server'ı çalıştır
    app.run_server(debug=False, use_reloader=False)


# -----------------------------------------------------
# Ana Program
# -----------------------------------------------------
def main():
    # Excel dosya yolunu belirtiyoruz
    file_path = r"C:\Users\Monster\Desktop\PROLAB 3 - GÜNCEL DATASET (1).xlsx"

    print(f"Veri yükleniyor: {file_path}")
    df = load_data(file_path)
    if df is None:
        print("Veri yüklenemedi. Program sonlandırılıyor.")
        sys.exit(1)

    # DataFrame'den GraphData oluşturma
    graph = create_graph(df)
    if graph is None or not graph.nodes:
        print("Graf oluşturulamadı. Program sonlandırılıyor.")
        sys.exit(1)

    print("Graf başarıyla oluşturuldu.")

    # Bazı istatistikler
    num_nodes = len(graph.nodes)
    num_edges = sum(len(node.edges) for node in graph.nodes.values()) // 2
    print(f"Toplam Düğüm Sayısı: {num_nodes}")
    print(f"Toplam Kenar Sayısı: {num_edges}")

    # Ana yazarları bulalım (ORCID dolu olanlar)
    main_author_names = set()
    for name_lower, node in graph.nodes.items():
        if node.author_position == 1:
            main_author_names.add(node.name)

    print("Ana yazar (ORCID dolu) isimleri:", main_author_names)

    # Her yazarın makale sayısını hesapla, ortalama makale sayısını bul
    paper_counts = {node.name: len(node.papers) for node in graph.nodes.values()}
    average_papers = sum(paper_counts.values()) / len(paper_counts) if paper_counts else 0

    # Yazarları dairesel yerleştirme
    positions = create_custom_circular_layout(graph)

    # Cytoscape'e uygun element listesi oluşturma
    cy_elements = create_cytoscape_elements(graph, paper_counts, average_papers, positions)

    # Tarayıcı otomatik açma
    threading.Timer(1, open_browser).start()

    # Dash uygulamasını çalıştır
    run_dash(graph, cy_elements, paper_counts, average_papers, main_author_names, {})

    print("Program sona erdi.")


def open_browser():
    """
    Localhost'ta çalışan Dash uygulamasını varsayılan tarayıcıda açar.
    """
    webbrowser.open_new("http://127.0.0.1:8050/")


if __name__ == "__main__":
    main()
