import sys
from IPython.display import HTML, display as ipy_display


class Display:
    '''A basic display.'''

    def display(self, game):
        if game.end == 2:
            self.win(game)
        elif game.end == 1:
            self.lose(game)
        else:
            self.show(game)

    def _display(self, game):
        print(game)

    def show(self, game):
        self._display(game)

    def win(self, game):
        self._display(game)
        print("You win! Score: %s" % game.score)

    def lose(self, game):
        self._display(game)
        print("You lose! Score: %s" % game.score)


class IPythonDisplay(Display):
    '''A better display for IPython (Jupyter) notebook environments.'''

    def __init__(self, display_size=40):
        self.display_size = display_size

    def _render(self, game):
        board = game.board
        html = '''<h1>Score: {}</h1>'''.format(game.score)
        table = '''<table style="border: 5px solid black;">{}</table>'''
        td = '''<td style="border:3px solid black; text-align:center;"
         width="%s" height="%s">{}</td>''' % (self.display_size, self.display_size)
        content = ''
        for row in range(game.size):
            content += '''<tr>'''
            for col in range(game.size):
                elem = int(board[row, col])
                content += td.format(elem if elem else "")
            content += '''</tr>'''
        html += table.format(content)
        return html

    def _display(self, game):
        if 'ipykernel' in sys.modules:
            source = self._render(game)
            ipy_display(HTML(source))
        else:
            print("Warning: since it's not in ipykernel, "
                  "it will show the command line version.")
            super()._display(game)
