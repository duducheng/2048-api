const app = new Vue({
    el: '#app',
    data: {
        score: 0,
        board: [],
        end: 0
    },
    created: function() {
        console.log("READY");
        this.$http.get('http://localhost:5005/board').then(
            this.updateResponse,
            console.log
        );
    },
    methods: {
        keyHandler: function(keycode) {
            // game: LEFT = 0, UP = 3, RIGHT = 2, DOWN = 1
            // keycode: LEFT = 37, UP = 38, RIGHT = 39, DOWN = 40
            const index = [37, 40, 39, 38].indexOf(keycode);
            console.log(keycode);
            console.log(index);
            // if index == -1, then AUTO!
            if (this.end == 0) {
                this.$http.post('http://localhost:5005/board', index).then(
                    this.updateResponse,
                    console.log
                );
            }
        },
        updateResponse: function(response) {
            this.board = response.body.board;
            this.score = response.body.score;
            this.end = response.body.end;
            this.direction = response.body.direction;
            this.control = response.body.control;
        }
    }
});


document.body.addEventListener('keydown', e => app.keyHandler(e.keyCode));