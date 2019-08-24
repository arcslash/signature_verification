import tornado.web.Application
import tornado.ioloop
from handler import Handler




def run_app():
    return tornado.web.Application([
        (r"/", Handler),
    ])

if __name__ == "__main__":
    app = run_app()
    app.listen(8888)
    tornado.ioloop.IOLoop.current().start()