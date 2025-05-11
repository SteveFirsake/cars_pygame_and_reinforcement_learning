import random
import sys

import pygame
from pygame import QUIT

from agent import CarAgent

pygame.init()

# background music
pygame.mixer.music.load("materials/background.wav")
pygame.mixer.music.play(-1)

# Initialize the joysticks.
pygame.joystick.init()

# Setting up FPS
FPS = 60
FramePerSec = pygame.time.Clock()

# Creating colors
BLUE = (0, 0, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

# Other Variables for use in the program
SCREEN_WIDTH = 393
SCREEN_HEIGHT = 597
SPEED = 5
SCORE = 0

# Setting up Fonts
font = pygame.font.SysFont("Verdana", 60)
font_small = pygame.font.SysFont("Verdana", 20)
game_over = font.render("Game Over", True, BLACK)

# static background - uncomment (DISPLAYSURF.blit(background, (0,0))) in Game Loop
# background = pygame.image.load("materials/AnimatedStreet.png")

# Create a white screen
DISPLAYSURF = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
DISPLAYSURF.fill(WHITE)
pygame.display.set_caption("Car Game")

# Initialize RL agent
agent = CarAgent(
    state_size=6, action_size=4
)  # 6 state features, 4 actions (up, down, left, right)


class Background:
    def __init__(self):
        global SPEED
        self.bgimage = pygame.image.load("materials/AnimatedStreet.png")
        self.rectBGimg = self.bgimage.get_rect()

        self.bgY1 = 0
        self.bgX1 = 0

        self.bgY2 = -self.rectBGimg.height
        self.bgX2 = 0

        self.movingUpSpeed = SPEED

    def update(self):
        self.bgY1 += self.movingUpSpeed
        self.bgY2 += self.movingUpSpeed
        if self.bgY1 >= self.rectBGimg.height:
            self.bgY1 = -self.rectBGimg.height
        if self.bgY2 >= self.rectBGimg.height:
            self.bgY2 = -self.rectBGimg.height

    def render(self):
        DISPLAYSURF.blit(self.bgimage, (self.bgX1, self.bgY1))
        DISPLAYSURF.blit(self.bgimage, (self.bgX2, self.bgY2))


class Enemy(pygame.sprite.Sprite):
    def __init__(self):
        super().__init__()
        self.image = pygame.image.load("materials/Enemy.png")
        self.surf = pygame.Surface((40, 90))
        self.rect = self.surf.get_rect(
            center=(random.randint(40, SCREEN_WIDTH - 40), 0)
        )

    def move(self):
        global SCORE
        self.rect.move_ip(0, SPEED)
        if self.rect.top > SCREEN_HEIGHT:
            SCORE += 1
            self.rect.top = 0
            self.rect.center = (random.randint(40, SCREEN_WIDTH - 40), 0)

    def reset(self):
        self.rect.top = 0
        self.rect.center = (random.randint(40, SCREEN_WIDTH - 40), 0)


class Player(pygame.sprite.Sprite):
    def __init__(self):
        super().__init__()
        self.image = pygame.image.load("materials/Player.png")
        self.surf = pygame.Surface((45, 95))
        self.rect = self.surf.get_rect(center=(153, 517))

    def move(self, action):
        # Action: 0=up, 1=down, 2=left, 3=right
        if action == 0 and self.rect.top > 0:  # Up
            self.rect.move_ip(0, -5)
        elif action == 1 and self.rect.bottom < SCREEN_HEIGHT:  # Down
            self.rect.move_ip(0, 5)
        elif action == 2 and self.rect.left > 0:  # Left
            self.rect.move_ip(-5, 0)
        elif action == 3 and self.rect.right < SCREEN_WIDTH:  # Right
            self.rect.move_ip(5, 0)

    def reset(self):
        self.rect.center = (153, 517)


def reset_game():
    global SCORE, SPEED
    SCORE = 0
    SPEED = 5
    P1.reset()
    E1.reset()


# Setting up Sprites
P1 = Player()
E1 = Enemy()

# background object
back_ground = Background()

# Creating Sprites Groups
enemies = pygame.sprite.Group()
enemies.add(E1)
all_sprites = pygame.sprite.Group()
all_sprites.add(P1)
all_sprites.add(E1)

# Adding a new User event
INC_SPEED = pygame.USEREVENT + 1
pygame.time.set_timer(INC_SPEED, 1000)

# Training loop
episode = 0
max_episodes = 1000
running = True

while running and episode < max_episodes:
    state = agent.get_state(P1, E1)
    total_reward = 0
    done = False

    while not done:
        # Get action from agent
        action = agent.get_action(state)

        # Move player based on agent's action
        P1.move(action)

        # Move enemy
        E1.move()

        # Get new state
        next_state = agent.get_state(P1, E1)

        # Calculate reward
        reward = 0
        if pygame.sprite.spritecollideany(P1, enemies):
            reward = -100  # Collision penalty
            done = True
            # Play crash sound
            pygame.mixer.Sound('materials/crash.wav').play()
            # Show game over screen
            DISPLAYSURF.fill(RED)
            DISPLAYSURF.blit(game_over, (23, 247))
            pygame.display.update()
            # Wait for a moment
            pygame.time.wait(1000)
            # Reset game state
            reset_game()
        else:
            reward = 1  # Small reward for surviving
            if E1.rect.top > SCREEN_HEIGHT:
                reward = 10  # Bonus for letting enemy pass

        # Train agent
        agent.train(state, action, reward, next_state, done)

        # Update state
        state = next_state
        total_reward += reward

        # Update display
        back_ground.update()
        back_ground.render()

        # Display score
        scores = font_small.render(f"SCORE: {SCORE}", True, BLACK)
        DISPLAYSURF.blit(scores, (10, 10))

        # Draw sprites
        for entity in all_sprites:
            DISPLAYSURF.blit(entity.image, entity.rect)

        # Check for quit
        for event in pygame.event.get():
            if event.type == QUIT:
                running = False
                done = True
            elif event.type == INC_SPEED:
                SPEED += 0.5

        pygame.display.update()
        FramePerSec.tick(FPS)

    # Reset game state for next episode
    if done:
        reset_game()
        episode += 1
        print(
            f"Episode {episode}, Total Reward: {total_reward}, Epsilon: {agent.epsilon:.2f}"
        )

# Save the trained model
agent.save("trained_car_agent.pkl")
pygame.quit()
sys.exit()
