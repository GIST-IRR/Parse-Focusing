import signal

class TrainingController:
    def __init__(self, handler=None) -> None:
        self.flag = False
        
        if handler is None:
            def signal_handler(signum, frame):
                self.flag = True
            handler = signal_handler

        signal.signal(signal.SIGINT, handler)

    def __call__(self, args):
        if self.flag:
            while True:
                # Check save directory
                prompt = \
    """Reserve: reserve termination of process after epoch N.
    Save: Save result of training now and terminate. 
    Exit: terminate process without saving. 
    What do you want to do? [reserve/save/exit]: """

                cmd = input(prompt)
                if cmd == 'reserve':
                    cmd = input(
                        "Do you want to reserve terminating training? [y/n]: "
                    )
                    cmd = input(f"Save to dir[{args.save_dir}]: ")
                    cmd = cmd.strip()
                elif cmd == 'save':
                    cmd = input("\b\bDo you want to save the model? [y/n]: ")
                elif cmd == 'exit':
                    print("Exit without saving.")
                    exit(0)
                else:
                    print("Invalid command.")
                    continue