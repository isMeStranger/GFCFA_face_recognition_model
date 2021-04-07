import os


def info():
    print("""

    FILE: project/code/Message.py

    PROJECT TITLE: Face Recog. Using Gabor Filters with The CFA

    This file contains the functions used to:

        *Show a message then return TRUE  for YES
                                and FALSE for NO
        

    Code by   :      Salar Adel Sabry
    Supervisor:  Mr. Haval Ismael Hussein
    """)
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


def yes_no_msg(msg):
    responses = ['y', 'Y', 'Yes', 'yes', 'YES']
    if str(input('\n'+msg+' (y/n): ')) in responses:
        return True
    else:
        return False


def create_dir(name):
    if not os.path.exists(name):
        os.makedirs(name)


if __name__ == '__main__':
    info()
    # test
    result = yes_no_msg('test message?')
    print('result', result)