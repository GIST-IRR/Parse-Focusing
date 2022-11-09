from .train_support import create_save_path

import functools
import shutil
import traceback

class Command:
    """If you want to use command decorator as a decorator, you just add decorator to the function that you want to use.
    But, if you want to  use command decorator as a function, you first call the function and then call the function that you want to use with the argument of the function that you called.

    Args:
        command (class or function): main function that you want to use. If you want to use class, you should add __call__ function to the class.
    """
    def __init__(self, copy_config=True, copy_code=True):
        self.copy_config = copy_config
        self.copy_code = copy_code

    def __call__(self, command):
        @functools.wraps()
        def wrapper(args):
            try:
                create_save_path(
                    args,
                    copy_config=self.copy_config,
                    copy_code=self.copy_code
                )
                result = command(args)
                return result
            except KeyboardInterrupt:
                while True:
                    print(f'Save dir: {args.save_dir}')
                    cmd = input('Do you want to save the model? [y/n]: ')
                    if cmd == 'y':
                        print("Log directory have been saved.")
                        break
                    elif cmd == 'n':
                        shutil.rmtree(args.save_dir)
                        print("You have successfully delete the created log directory.")
                        break
                    else:
                        print("Please enter 'y' or 'n'.")
                        continue
            except Exception:
                traceback.print_exc()
                while True:
                    print(f'Save dir: {args.save_dir}')
                    cmd = input('Do you want to save the model? [y/n]: ')
                    if cmd == 'y':
                        print("Log directory have been saved.")
                        break
                    elif cmd == 'n':
                        shutil.rmtree(args.save_dir)
                        print("You have successfully delete the created log directory.")
                        break
                    else:
                        print("Please enter 'y' or 'n'.")
                        continue
        return wrapper

    