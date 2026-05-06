from pynput import mouse, keyboard
import threading
import time

clicking = False

def click_loop():
    global clicking
    m = mouse.Controller()
    while True:
        if clicking:
            m.click(mouse.Button.left)
            time.sleep(0.03)  # velocidade (menor = mais rápido)
        else:
            time.sleep(0.01)

def on_press(key):
    global clicking
    try:
        if key.char == 'f':  # tecla que ativa
            clicking = True
    except:
        pass

def on_release(key):
    global clicking
    try:
        if key.char == 'f':
            clicking = False
    except:
        pass

# thread de clique
threading.Thread(target=click_loop, daemon=True).start()

# listener de teclado
with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
    listener.join()