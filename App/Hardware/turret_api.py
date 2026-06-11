# Usart Library
import serial
import time as t

ERROR_CODES = {
    "11214": "Motor timeout (turret failed to reach position)",
    "10101": "Command format error",
    "10102": "Parameter error",
    "10103": "Command not executable",
}

NUM_OBJECTIVES = 5

class turret:
    """
    API class for controlling BX-REMCB turret controller
    """

    def __init__(self, port: int):
        """
        Initialize the serial port and log in to the controller
        """
        print("Starting olympus controller...")
        
        # Initialize serial connection
        self.Usart = serial.Serial(
            port=f"COM{port}",
            baudrate=19200,                     # BX-REMCB default baudrate
            bytesize=serial.EIGHTBITS,          # 8 data bits
            parity=serial.PARITY_EVEN,          # Even parity
            stopbits=serial.STOPBITS_TWO,       # 2 stop bits
            timeout=1                           # 1 s read timeout
        )
        
        '''
        # Check CTS status
        if self.Usart.getCTS():
            print("CTS asserted (ready to receive)")
        else:
            print("CTS de-asserted (not ready)")
        '''
        
        t.sleep(1)
        
        # Log in to the controller
        self.Usart.write('1LOG IN\r\n'.encode())
        t.sleep(0.5)
        current_response = self.Usart.readline()
        
        if current_response == b'1LOG +\r\n':
            print("Connected to olympus controller!")
        else:
            print(f"Connection failed. Response: {current_response}")

        self.is_moving = False

    def send_command(self, command, timeout=5):

        self.Usart.reset_input_buffer()

        full_cmd = f"{command}\r\n".encode()

        #print(f"Sending: {command}")
        self.Usart.write(full_cmd)

        start = t.time()

        while True:

            if t.time() - start > timeout:
                raise TimeoutError(f"Timeout waiting for response to {command}")

            response = self.Usart.readline()

            if response:

                response = response.decode().strip()
                #print(f"Received: {response}")

                # ERROR
                if "!" in response:
                    raise Exception(f"Controller error: {response}")

                # SUCCESS ACK
                if "+" in response:
                    return response

                # QUERY RESPONSE (value returned)
                if "?" in command:
                    return response
            
    def check_if_log_in(self):
        self.Usart.write("1LOG?\r\n".encode())
        t.sleep(0.5)
        response = self.Usart.readline()
        print(response)
        
        '''
        # Check for valid response indicating logged in status
        if response and b'1LOG IN' in response:
            print("Controller is logged in")
            return True
        else:
            print("Controller is not logged in")
            return False
        '''
    
    def turn_to_position(self, value, output=False):

        if not 1 <= value <= NUM_OBJECTIVES:
            raise ValueError("Turret position must be 1–5")

        if self.is_moving:
            if output: print("Turret is already moving. Command ignored.")
            return

        self.is_moving = True

        try:
            if output: print(f"Moving turret to position {value}")

            self.send_command(f"1OB {value}")

            while True:
                pos = self.check_position()
                if pos == value:
                    break
                t.sleep(0.1)

            if output: print("Move complete")

        finally:
            self.is_moving = False
    
    def check_position(self, output=False):

        response = self.send_command("1OB?")

        try:
            parts = response.split()
            position = int(parts[1])
            if output: print("Current position:", position)
            return position
        except:
            if output: print("Unexpected response:", response)
            return None
    
    def close(self):
        try:
            # Log out
            self.Usart.write('1LOG OUT\r\n'.encode())

            t.sleep(0.5)

            logout_response = self.Usart.readline()
            print(f"Logout response: {logout_response}")
            
            # Close serial port
            self.Usart.close()
            print("Serial port closed")
        except Exception as e:
            print(f"Error during close: {e}")