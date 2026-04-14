use std::io;

use crossterm::ExecutableCommand;
use crossterm::event::{self, Event, KeyCode, KeyEventKind};
use crossterm::terminal::{
    EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode,
};
use ratatui::Terminal;
use ratatui::backend::{Backend, CrosstermBackend};
use ratatui::layout::{Constraint, Direction, Layout};
use ratatui::style::{Style, Stylize};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Paragraph, Wrap};

use crate::Cortex;

const MAX_GENERATED_TOKENS: usize = 200;

struct App {
    cortex: Cortex,
    transcript: Vec<Line<'static>>,
    input: String,
    quit: bool,
}

pub fn run(cortex: Cortex) -> io::Result<()> {
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    stdout.execute(EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    let mut app = App {
        cortex,
        transcript: vec![Line::from(Span::raw(
            "cortex chat. type a prompt, enter to send, esc to quit.",
        ))],
        input: String::new(),
        quit: false,
    };

    let result = event_loop(&mut terminal, &mut app);

    disable_raw_mode()?;
    terminal.backend_mut().execute(LeaveAlternateScreen)?;
    terminal.show_cursor()?;
    result
}

fn event_loop<B: Backend>(terminal: &mut Terminal<B>, app: &mut App) -> io::Result<()> {
    while !app.quit {
        terminal.draw(|f| render(f, app))?;
        if event::poll(std::time::Duration::from_millis(100))?
            && let Event::Key(key) = event::read()?
        {
            if key.kind != KeyEventKind::Press {
                continue;
            }
            match key.code {
                KeyCode::Esc => app.quit = true,
                KeyCode::Enter => submit(app),
                KeyCode::Backspace => {
                    app.input.pop();
                }
                KeyCode::Char(c) => app.input.push(c),
                _ => {}
            }
        }
    }
    Ok(())
}

fn submit(app: &mut App) {
    let prompt = std::mem::take(&mut app.input);
    if prompt.is_empty() {
        return;
    }
    app.transcript.push(Line::from(Span::styled(
        format!("> {}", prompt),
        Style::new().bold(),
    )));
    let completion = app.cortex.generate(&prompt, MAX_GENERATED_TOKENS);
    app.transcript.push(Line::from(Span::raw(completion)));
}

fn render(frame: &mut ratatui::Frame, app: &App) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Min(1), Constraint::Length(3)])
        .split(frame.area());

    let transcript = Paragraph::new(app.transcript.clone())
        .wrap(Wrap { trim: false })
        .block(Block::default().borders(Borders::ALL).title("transcript"));
    frame.render_widget(transcript, chunks[0]);

    let input = Paragraph::new(app.input.as_str())
        .block(Block::default().borders(Borders::ALL).title("input (esc quits)"));
    frame.render_widget(input, chunks[1]);
}
