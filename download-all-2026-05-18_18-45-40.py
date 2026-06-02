#!/usr/bin/python
"""
Script de bulk download do ASF adaptado para autenticação via token Earthdata.
Importa o token do módulo TokenAFS (variável TOKEN_AFS).
Uso:
    python ./download-all-2026-05-18_18-45-40.py [--insecure] [arquivo.metalink|arquivo.csv ...]
Compatibilidade: Python >= 3.6
"""
import sys, csv, os, os.path, tempfile, shutil, re, time, ssl, signal, socket
from urllib.parse import urlparse
from urllib.request import build_opener, install_opener, Request, urlopen
from urllib.request import HTTPHandler, HTTPSHandler
from urllib.error import HTTPError, URLError
import xml.etree.ElementTree as ET
from io import StringIO

# Importa o token (ajuste o caminho se necessário)
try:
    from .TokenAFS import TOKEN_AFS
except ImportError:
    # Fallback para importação direta se rodar como script standalone
    from TokenAFS import TOKEN_AFS

abort = False

def signal_handler(sig, frame):
    global abort
    sys.stderr.write("\n > Caught Signal. Exiting!\n")
    abort = True
    raise SystemExit

class bulk_downloader:
    def __init__(self):
        self.files = [ "https://datapool.asf.alaska.edu/RTC_HI_RES/A3/AP_26256_FBS_F6990_RT1.zip" ]
        self.token = TOKEN_AFS
        if not self.token or len(self.token) < 10:
            print("ERROR: Token inválido. Verifique o arquivo TokenAFS.py")
            exit(-1)

        if os.access(os.getcwd(), os.W_OK) is False:
            print(f"WARNING: Cannot write to current path! Check permissions for {os.getcwd()}")
            exit(-1)

        self.context = {}

        # Processa argumentos da linha de comando
        if len(sys.argv) > 1:
            download_files = []
            for arg in sys.argv[1:]:
                if arg == '--insecure':
                    try:
                        ctx = ssl.create_default_context()
                        ctx.check_hostname = False
                        ctx.verify_mode = ssl.CERT_NONE
                        self.context['context'] = ctx
                    except AttributeError:
                        pass
                elif arg.endswith('.metalink') or arg.endswith('.csv'):
                    if os.path.isfile(arg):
                        if arg.endswith('.metalink'):
                            new_files = self.process_metalink(arg)
                        else:
                            new_files = self.process_csv(arg)
                        if new_files:
                            download_files.extend(new_files)
                    else:
                        print(f" > Cannot find input file: {arg}")
                else:
                    print(f" > Argument '{arg}' ignored.")
            if download_files:
                self.files = download_files
                print(f" > Processing {len(self.files)} downloads from command-line arguments.")
            else:
                print(" > No valid files found in arguments, using hardcoded list.")

        self.total_bytes = 0
        self.total_time = 0
        self.cnt = 0
        self.success = []
        self.failed = []
        self.skipped = []

    # ------------------------------------------------------------
    # Substitui todo o sistema de cookies por token Bearer
    # ------------------------------------------------------------
    def download_file_with_token(self, url, file_count, total):
        download_file = os.path.basename(url).split('?')[0]

        # Verifica se o arquivo já existe e é completo
        if os.path.isfile(download_file):
            try:
                request = Request(url)
                request.add_header('Authorization', f'Bearer {self.token}')
                local_size = os.path.getsize(download_file)
                remote_size = self.get_total_size(urlopen(request))
                if remote_size and (local_size + local_size*0.01) > remote_size > (local_size - local_size*0.01):
                    print(f" > File {download_file} exists, skipping.")
                    return None, None
                else:
                    print(f" > {download_file} incomplete, redownloading.")
                    os.remove(download_file)
            except Exception as e:
                print(f" > HEAD request failed: {e}")

        # Tenta o download
        try:
            request = Request(url)
            request.add_header('Authorization', f'Bearer {self.token}')
            response = urlopen(request, timeout=30)

            while response.getcode() == 202:
                print(" > Waiting for burst extraction service...")
                time.sleep(5)
                request = Request(url)
                request.add_header('Authorization', f'Bearer {self.token}')
                response = urlopen(request, timeout=30)

            # Redirecionamento simples (se não for para URS)
            if response.geturl() != url:
                if 'urs.earthdata.nasa.gov' in response.geturl():
                    print(" > Redirected to URS – token may be invalid or expired.")
                    return False, None
                print(f" > Redirected to {response.geturl()}")

            print(f"({file_count}/{total}) Downloading {url}")
            content_disp = response.headers.get('Content-Disposition')
            if content_disp:
                match = re.findall(r"filename=(\S+)", content_disp)
                if match:
                    download_file = match[0]

            tf = tempfile.NamedTemporaryFile(mode='w+b', delete=False, dir='.')
            self.chunk_read(response, tf, report_hook=self.chunk_report)
            sys.stdout.write('\n')
            tempfile_name = tf.name
            tf.close()

        except HTTPError as e:
            print(f"HTTP Error {e.code}: {url}")
            if e.code == 401:
                print(" > Token inválido ou expirado.")
            elif e.code == 403:
                print(" > Acesso negado. Verifique se aceitou os termos de uso no ASF Vertex.")
            return False, None
        except (URLError, socket.timeout, ssl.CertificateError) as e:
            print(f"Error: {e}")
            return False, None

        # Finaliza o arquivo
        shutil.copy(tempfile_name, download_file)
        os.remove(tempfile_name)
        file_size = self.get_total_size(response)
        actual_size = os.path.getsize(download_file)
        if file_size is None:
            file_size = actual_size
        return actual_size, file_size

    # Os métodos auxiliares chunk_read, chunk_report, get_total_size, process_metalink, process_csv,
    # download_files, is_good_download e print_summary permanecem os mesmos, apenas a chamada
    # a download_file_with_cookiejar é substituída por download_file_with_token.

    def chunk_report(self, bytes_so_far, file_size):
        if file_size is not None:
            percent = round(float(bytes_so_far)/file_size*100, 2)
            sys.stdout.write(f" > Downloaded {bytes_so_far} of {file_size} bytes ({percent:0.2f}%)\r")
        else:
            sys.stdout.write(f" > Downloaded {bytes_so_far} of unknown Size\r")

    def chunk_read(self, response, local_file, chunk_size=8192, report_hook=None):
        file_size = self.get_total_size(response)
        bytes_so_far = 0
        while True:
            try:
                chunk = response.read(chunk_size)
            except Exception:
                sys.stdout.write("\n > Error reading data.\n")
                break
            try:
                local_file.write(chunk)
            except TypeError:
                local_file.write(chunk.decode(local_file.encoding))
            bytes_so_far += len(chunk)
            if not chunk:
                break
            if report_hook:
                report_hook(bytes_so_far, file_size)
        return bytes_so_far

    def get_total_size(self, response):
        try:
            return int(response.info().getheader('Content-Length').strip())
        except AttributeError:
            try:
                return int(response.getheader('Content-Length').strip())
            except:
                return None

    def process_metalink(self, ml_file):
        print(f"Processing metalink: {ml_file}")
        with open(ml_file, 'r') as f:
            xml = f.read()
        it = ET.iterparse(StringIO(xml))
        for _, el in it:
            if '}' in el.tag:
                el.tag = el.tag.split('}', 1)[1]
        root = it.root
        urls = []
        for dl in root.find('files'):
            urls.append(dl.find('resources').find('url').text)
        return urls if urls else None

    def process_csv(self, csv_file):
        print(f"Processing csv: {csv_file}")
        urls = []
        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                urls.append(row['URL'])
        return urls if urls else None

    def download_files(self):
        for file_name in self.files:
            if abort:
                raise SystemExit
            self.cnt += 1
            start = time.time()
            size, total_size = self.download_file_with_token(file_name, self.cnt, len(self.files))
            end = time.time()
            if size is None:
                self.skipped.append(file_name)
            elif self.is_good_download(total_size, size):
                elapsed = max(end - start, 1)
                rate = (size/1024**2)/elapsed
                print(f"Downloaded {size}b in {elapsed:.2f}s, rate: {rate:.2f}MB/s")
                self.total_bytes += size
                self.total_time += elapsed
                self.success.append({'file': file_name, 'size': size})
            else:
                print(f"Problem downloading {file_name}")
                self.failed.append(file_name)

    def is_good_download(self, total_size, size):
        if size is False or total_size is None:
            return False
        return (total_size < (size + size*0.01)) and (total_size > (size - size*0.01))

    def print_summary(self):
        print("\n\nDownload Summary")
        print("-" * 80)
        print(f"  Success: {len(self.success)} files, {self.total_bytes} bytes")
        for s in self.success:
            print(f"           - {s['file']}  {s['size']/1024**2:.2f}MB")
        if self.failed:
            print(f"  Failures: {len(self.failed)}")
            for f in self.failed:
                print(f"          - {f}")
        if self.skipped:
            print(f"  Skipped: {len(self.skipped)}")
            for f in self.skipped:
                print(f"          - {f}")
        if self.success:
            rate = (self.total_bytes/1024**2) / self.total_time if self.total_time else 0
            print(f"  Average rate: {rate:.2f}MB/s")
        print("-" * 80)

if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    downloader = bulk_downloader()
    downloader.download_files()
    downloader.print_summary()